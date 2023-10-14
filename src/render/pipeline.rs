/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::{
	allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayoutCreateInfo, PersistentDescriptorSet,
	WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::{Format, NumericType};
use vulkano::pipeline::{
	layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, PipelineLayout,
	PipelineShaderStageCreateInfo
};
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState},
	depth_stencil::{CompareOp, DepthState, DepthStencilState},
	input_assembly::{InputAssemblyState, PrimitiveTopology},
	multisample::MultisampleState,
	rasterization::{CullMode, RasterizationState},
	vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState},
	viewport::ViewportState,
	subpass::PipelineRenderingCreateInfo,
	GraphicsPipelineCreateInfo,
};
use vulkano::image::sampler::Sampler;
use vulkano::shader::{EntryPoint, ShaderInterfaceEntryType, ShaderModule};

use crate::GenericEngineError;

pub struct Pipeline
{
	descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
	pipeline: Arc<GraphicsPipeline>,

	/// similar pipeline, except for the fragment shader being capable of processing Order-Independent Transparency
	transparency_pipeline: Option<Arc<GraphicsPipeline>>,
}
impl Pipeline
{
	pub fn new(
		primitive_topology: PrimitiveTopology,
		vs_filename: String,
		fs_info: Option<(String, ColorBlendState)>,
		fs_transparency_info: Option<(String, PipelineRenderingCreateInfo)>,
		samplers: Vec<(usize, u32, Arc<Sampler>)>, // set: usize, binding: u32, sampler: Arc<Sampler>
		rendering_info: PipelineRenderingCreateInfo,
		depth_op: CompareOp,
		depth_write: bool,
		descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
	) -> Result<Self, GenericEngineError>
	{
		let vk_dev = descriptor_set_allocator.device().clone();

		let (vs, vertex_input_state) = load_spirv_vertex(vk_dev.clone(), &Path::new("shaders").join(vs_filename))?;

		let mut input_assembly_state = InputAssemblyState::new().topology(primitive_topology);
		if primitive_topology == PrimitiveTopology::TriangleStrip {
			input_assembly_state = input_assembly_state.primitive_restart_enable();
		}

		let mut depth_stencil_state = DepthStencilState {
			depth: Some(DepthState {
				write_enable: depth_write,
				compare_op: depth_op,
			}),
			..Default::default()
		};
		let mut stages = Vec::with_capacity(5);
		stages.push(PipelineShaderStageCreateInfo::new(get_entry_point(&vs, "main")?));
		
		// load fragment shader (optional)
		let mut transparency_pipeline = None;
		let mut color_blend_state = None;
		if let Some((fs_filename, blend_state)) = fs_info {
			// load the fragment shader for the opaque pass
			let fs = load_spirv(vk_dev.clone(), &Path::new("shaders").join(fs_filename))?;
			stages.push(PipelineShaderStageCreateInfo::new(get_entry_point(&fs, "main")?));

			// use a different fragment shader and pipeline for OIT
			if let Some((ft, ft_rendering_info)) = fs_transparency_info {
				let mut wboit_accum_blend = ColorBlendState::new(2);
				wboit_accum_blend.attachments[0].blend = Some(AttachmentBlend {
					alpha_blend_op: BlendOp::Add,
					..AttachmentBlend::additive()
				});
				wboit_accum_blend.attachments[1].blend = Some(AttachmentBlend {
					color_blend_op: BlendOp::Add,
					src_color_blend_factor: BlendFactor::Zero,
					dst_color_blend_factor: BlendFactor::OneMinusSrcColor,
					..AttachmentBlend::ignore_source()
				});

				let fs_transparency = load_spirv(vk_dev.clone(), &Path::new("shaders").join(ft))?;

				let transparency_stages = vec![
					PipelineShaderStageCreateInfo::new(get_entry_point(&vs, "main")?),
					PipelineShaderStageCreateInfo::new(get_entry_point(&fs_transparency, "main")?),
				];

				let mut transparency_depth_stencil_state = depth_stencil_state.clone();
				// WBOIT needs depth write to be disabled
				transparency_depth_stencil_state.depth.as_mut().unwrap().write_enable = false;

				let mut transparency_set_layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&transparency_stages);
				pipeline_sampler_setup(transparency_set_layout_info.set_layouts.as_mut_slice(), &samplers);
				let transparency_layout = PipelineLayout::new(
					vk_dev.clone(),
					transparency_set_layout_info.into_pipeline_layout_create_info(vk_dev.clone())?
				)?;
				let mut transparency_pipeline_info = GraphicsPipelineCreateInfo {
					stages: transparency_stages.into(),
					vertex_input_state: Some(vertex_input_state.clone()),
					input_assembly_state: Some(input_assembly_state.clone()),
					viewport_state: Some(ViewportState::default()),
					rasterization_state: Some(RasterizationState::new().cull_mode(CullMode::Back)),
					multisample_state: Some(MultisampleState::default()),
					depth_stencil_state: Some(transparency_depth_stencil_state),
					color_blend_state: Some(wboit_accum_blend),
					dynamic_state: [ DynamicState::Viewport ].into_iter().collect(),
					subpass: Some(ft_rendering_info.into()),	
					..GraphicsPipelineCreateInfo::layout(transparency_layout)
				};

				let built_transparency_pipeline = GraphicsPipeline::new(vk_dev.clone(), None, transparency_pipeline_info)?;
				print_pipeline_descriptors_info(built_transparency_pipeline.as_ref());
				transparency_pipeline = Some(built_transparency_pipeline);
			}

			color_blend_state = Some(blend_state);
		} 

		let mut pl_set_layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
		pipeline_sampler_setup(pl_set_layout_info.set_layouts.as_mut_slice(), &samplers);
		let pipeline_layout = PipelineLayout::new(
			vk_dev.clone(), 
			pl_set_layout_info.into_pipeline_layout_create_info(vk_dev.clone())?
		)?;
		let mut pipeline_info = GraphicsPipelineCreateInfo {
			stages: stages.into(),
			vertex_input_state: Some(vertex_input_state),
			input_assembly_state: Some(input_assembly_state),
			viewport_state: Some(ViewportState::default()),
			rasterization_state: Some(RasterizationState::new().cull_mode(CullMode::Back)),
			multisample_state: Some(MultisampleState::default()),
			depth_stencil_state: Some(depth_stencil_state.clone()),
			color_blend_state,
			dynamic_state: [ DynamicState::Viewport ].into_iter().collect(),
			subpass: Some(rendering_info.into()),
			..GraphicsPipelineCreateInfo::layout(pipeline_layout)
		};
	
		let pipeline = GraphicsPipeline::new(vk_dev.clone(), None, pipeline_info)?;
		print_pipeline_descriptors_info(pipeline.as_ref());

		Ok(Pipeline {
			descriptor_set_allocator,
			pipeline,
			transparency_pipeline,
		})
	}

	/// Create a pipeline from a YAML pipeline configuration file.
	pub fn new_from_yaml(
		yaml_filename: &str,
		rendering_info: PipelineRenderingCreateInfo,
		transparency_rendering: Option<PipelineRenderingCreateInfo>,
		tex_sampler: Arc<Sampler>,
		descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
	) -> Result<Self, GenericEngineError>
	{
		log::info!("Loading pipeline definition file '{}'...", yaml_filename);

		let yaml_reader = File::open(Path::new("shaders").join(yaml_filename))?;
		let deserialized: PipelineConfig = serde_yaml::from_reader(yaml_reader)?;
		let generated_samplers = deserialized
			.samplers
			.iter()
			.map(|sampler_config| (sampler_config.set, sampler_config.binding, tex_sampler.clone()))
			.collect();

		let mut color_blend_state = ColorBlendState::new(rendering_info.color_attachment_formats.len().try_into().unwrap());
		if deserialized.alpha_blending {
			color_blend_state = color_blend_state.blend_alpha();
		}

		let fs_info = deserialized.fragment_shader.map(|fs| (fs, color_blend_state));
		let fs_transparency_info = deserialized
			.fragment_shader_transparency
			.map(|fst| (fst, transparency_rendering.unwrap()));

		let depth_op = deserialized
			.always_pass_depth_test
			.then_some(CompareOp::Always)
			.unwrap_or(CompareOp::Less);

		Pipeline::new(
			deserialized.primitive_topology,
			deserialized.vertex_shader,
			fs_info,
			fs_transparency_info,
			generated_samplers,
			rendering_info,
			depth_op,
			true,
			descriptor_set_allocator,
		)
	}

	pub fn bind<L>(&self, command_buffer: &mut AutoCommandBufferBuilder<L>) -> Result<(), GenericEngineError>
	{
		command_buffer.bind_pipeline_graphics(self.pipeline.clone())?;
		Ok(())
	}
	pub fn bind_transparency<L>(&self, command_buffer: &mut AutoCommandBufferBuilder<L>) -> Result<(), GenericEngineError>
	{
		command_buffer.bind_pipeline_graphics(self.transparency_pipeline.clone().ok_or(TransparencyNotEnabled)?)?;
		Ok(())
	}

	pub fn layout(&self) -> Arc<PipelineLayout>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self.pipeline.as_ref();
		pipeline_ref.layout().clone()
	}

	pub fn layout_transparency(&self) -> Result<Arc<PipelineLayout>, TransparencyNotEnabled>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self
			.transparency_pipeline
			.as_ref()
			.ok_or(TransparencyNotEnabled)?
			.as_ref();
		Ok(pipeline_ref.layout().clone())
	}

	/// Create a new persistent descriptor set for use with the descriptor set slot at `set_number`, writing `writes`
	/// into the descriptor set.
	pub fn new_descriptor_set(
		&self,
		set_number: usize,
		writes: impl IntoIterator<Item = WriteDescriptorSet>,
	) -> Result<Arc<PersistentDescriptorSet>, GenericEngineError>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self.pipeline.as_ref();
		let set_layout = pipeline_ref
			.layout()
			.set_layouts()
			.get(set_number)
			.ok_or(format!("Pipeline::new_descriptor_set: invalid descriptor set index {}", set_number))?
			.clone();
		Ok(PersistentDescriptorSet::new(
			&self.descriptor_set_allocator,
			set_layout,
			writes,
			[]
		)?)
	}
}

#[derive(Debug)]
pub struct TransparencyNotEnabled;
impl Error for TransparencyNotEnabled {}
impl std::fmt::Display for TransparencyNotEnabled
{
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
	{
		write!(f, "this pipeline hasn't been set up with a fragment shader for OIT")
	}
}

#[derive(Deserialize)]
struct PipelineSamplerConfig
{
	set: usize,
	binding: u32,
	mag_linear: bool,
}

#[derive(Deserialize)]
struct PipelineConfig
{
	vertex_shader: String,
	fragment_shader: Option<String>,
	fragment_shader_transparency: Option<String>,

	#[serde(default)]
	always_pass_depth_test: bool,

	#[serde(default)]
	alpha_blending: bool,

	#[serde(with = "PrimitiveTopologyDef")]
	primitive_topology: PrimitiveTopology,

	#[serde(default)]
	samplers: Vec<PipelineSamplerConfig>,
	//#[serde(default)]
	//attachments: Vec<PipelineBlendState>,
}

// copy of `vulkano::pipeline::graphics::input_assembly::PrimitiveTopology` so we can more directly (de)serialize it
#[derive(Deserialize)]
#[serde(remote = "PrimitiveTopology")]
enum PrimitiveTopologyDef
{
	PointList,
	LineList,
	LineStrip,
	TriangleList,
	TriangleStrip,
	TriangleFan,
	LineListWithAdjacency,
	LineStripWithAdjacency,
	TriangleListWithAdjacency,
	TriangleStripWithAdjacency,
	PatchList,
}

fn get_entry_point(
	shader_module: &Arc<ShaderModule>,
	entry_point_name: &str,
) -> Result<EntryPoint, GenericEngineError>
{
	shader_module
		.entry_point(entry_point_name)
		.ok_or(format!("No entry point called '{}' found in SPIR-V module!", entry_point_name).into())
}

fn load_spirv(device: Arc<vulkano::device::Device>, path: &Path) -> Result<Arc<ShaderModule>, GenericEngineError>
{
	let print_file_name = path
		.file_name()
		.and_then(|f| f.to_str())
		.unwrap_or("[invalid file name in path]");
	log::info!("Loading shader file '{}'...", print_file_name);
	let spv_data = std::fs::read(path)?;
	Ok(unsafe { ShaderModule::from_bytes(device, &spv_data) }?)
}

/// Load the SPIR-V file, and also automatically determine the given vertex shader's vertex inputs using information from the
/// SPIR-V file.
fn load_spirv_vertex(
	device: Arc<vulkano::device::Device>,
	path: &Path,
) -> Result<(Arc<ShaderModule>, VertexInputState), GenericEngineError>
{
	let shader_module = load_spirv(device, path)?;
	let vertex_input_state = shader_module
		.entry_point("main")
		.ok_or("No valid 'main' entry point in SPIR-V module!")?
		.info()
		.input_interface
		.elements()
		.iter()
		.fold(VertexInputState::new(), |accum, input| {
			let binding = input.location;
			let format = format_from_interface_type(&input.ty);
			let stride = format.components().iter().fold(0, |acc, c| acc + (*c as u32)) / 8;
			accum
				.binding(
					binding,
					VertexInputBindingDescription {
						stride,
						input_rate: VertexInputRate::Vertex,
					},
				)
				.attribute(
					binding,
					VertexInputAttributeDescription {
						binding,
						format,
						offset: 0,
					},
				)
		});

	log::debug!("automatically generated VertexInputState: {:#?}", &vertex_input_state);

	Ok((shader_module, vertex_input_state))
}

fn format_from_interface_type(ty: &ShaderInterfaceEntryType) -> Format
{
	let possible_formats = match ty.base_type {
		NumericType::Float => [
			Format::R32_SFLOAT,
			Format::R32G32_SFLOAT,
			Format::R32G32B32_SFLOAT,
			Format::R32G32B32A32_SFLOAT,
		],
		NumericType::Int => [
			Format::R32_SINT,
			Format::R32G32_SINT,
			Format::R32G32B32_SINT,
			Format::R32G32B32A32_SINT,
		],
		NumericType::Uint => [
			Format::R32_UINT,
			Format::R32G32_UINT,
			Format::R32G32B32_UINT,
			Format::R32G32B32A32_UINT,
		],
	};
	let format_index = (ty.num_components - 1) as usize;
	possible_formats[format_index]
}

fn pipeline_sampler_setup(sets: &mut [DescriptorSetLayoutCreateInfo], samplers: &Vec<(usize, u32, Arc<Sampler>)>)
{
	for (set_i, binding_i, sampler) in samplers {
		match sets.get_mut(*set_i) {
			Some(s) => match s.bindings.get_mut(binding_i) {
				Some(b) => b.immutable_samplers = vec![sampler.clone()],
				None => log::warn!("Binding {} doesn't exist in set {}, ignoring!", binding_i, set_i),
			},
			None => log::warn!("Set index {} for sampler is out of bounds, ignoring!", set_i),
		}
	}
}

fn print_pipeline_descriptors_info(pipeline: &dyn vulkano::pipeline::Pipeline)
{
	log::debug!("Built pipeline with descriptors:");
	for ((set, binding), req) in pipeline.descriptor_binding_requirements() {
		let desc_count_string = match req.descriptor_count {
			Some(count) => format!("{}x", count),
			None => "runtime-sized array of".into(),
		};

		log::debug!(
			"set {}, binding {}: {} {:?}",
			set,
			binding,
			desc_count_string,
			req.descriptor_types
		);
	}
}
