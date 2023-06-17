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
use vulkano::format::Format;
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState},
	depth_stencil::{CompareOp, DepthState, DepthStencilState},
	input_assembly::{InputAssemblyState, PrimitiveTopology},
	rasterization::{CullMode, RasterizationState},
	vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState},
	viewport::ViewportState,
};
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, StateMode};
use vulkano::render_pass::Subpass;
use vulkano::sampler::Sampler;
use vulkano::shader::{EntryPoint, ShaderInterfaceEntryType, ShaderModule, ShaderScalarType};

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
		fs_transparency_info: Option<(String, Subpass)>,
		samplers: Vec<(usize, u32, Arc<Sampler>)>, // set: usize, binding: u32, sampler: Arc<Sampler>
		subpass: Subpass,
		depth_op: CompareOp,
		depth_write: bool,
		descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
	) -> Result<Self, GenericEngineError>
	{
		let vk_dev = subpass.render_pass().device().clone();

		let (vs, vertex_input_state) = load_spirv_vertex(vk_dev.clone(), &Path::new("shaders").join(vs_filename))?;

		let mut input_assembly_state = InputAssemblyState::new().topology(primitive_topology);
		if primitive_topology == PrimitiveTopology::TriangleStrip {
			input_assembly_state = input_assembly_state.primitive_restart_enable();
		}

		let mut depth_stencil_state = DepthStencilState {
			depth: Some(DepthState {
				enable_dynamic: false,
				write_enable: StateMode::Fixed(depth_write),
				compare_op: StateMode::Fixed(depth_op),
			}),
			..Default::default()
		};

		// do some building
		let mut pipeline_builder = GraphicsPipeline::start()
			.vertex_shader(get_entry_point(&vs, "main")?, ())
			.vertex_input_state(vertex_input_state)
			.input_assembly_state(input_assembly_state)
			.viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
			.rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
			.depth_stencil_state(depth_stencil_state.clone())
			.render_pass(subpass);

		// load fragment shader (optional)
		let fs;
		let fs_transparency;
		let mut builder_transparency = None;
		if let Some((fs_filename, blend_state)) = fs_info {
			fs = load_spirv(vk_dev.clone(), &Path::new("shaders").join(fs_filename))?;
			pipeline_builder = pipeline_builder.fragment_shader(get_entry_point(&fs, "main")?, ());

			// use a different fragment shader for OIT
			if let Some((ft, ft_subpass)) = fs_transparency_info {
				let mut wboit_accum_blend = ColorBlendState::new(2);
				wboit_accum_blend.attachments[0].blend = Some(AttachmentBlend {
					alpha_op: BlendOp::Add,
					..AttachmentBlend::additive()
				});
				wboit_accum_blend.attachments[1].blend = Some(AttachmentBlend {
					color_op: BlendOp::Add,
					color_source: BlendFactor::Zero,
					color_destination: BlendFactor::OneMinusSrcColor,
					..AttachmentBlend::ignore_source()
				});

				fs_transparency = load_spirv(vk_dev.clone(), &Path::new("shaders").join(ft))?;

				// WBOIT needs depth write to be disabled
				depth_stencil_state.depth.as_mut().unwrap().write_enable = StateMode::Fixed(false);

				builder_transparency = Some(
					pipeline_builder
						.clone()
						.depth_stencil_state(depth_stencil_state)
						.fragment_shader(get_entry_point(&fs_transparency, "main")?, ())
						.color_blend_state(wboit_accum_blend)
						.render_pass(ft_subpass),
				);
			} else {
				// only enable color blending for the basic fragment shader if there isn't a separate transparency shader
				pipeline_builder = pipeline_builder.color_blend_state(blend_state);
			}
		}

		// build pipeline with immutable samplers, if it needs any
		let pipeline = pipeline_builder.with_auto_layout(vk_dev.clone(), |sets| pipeline_sampler_setup(sets, &samplers))?;
		print_pipeline_descriptors_info(pipeline.as_ref());

		/*let transparency_pipeline = builder_transparency
		.map(|bt| -> Result<Arc<GraphicsPipeline>, GenericEngineError> {
			let pipeline_ref: &dyn vulkano::pipeline::Pipeline = pipeline.as_ref();
			let layout = pipeline_ref.layout().clone();
			Ok(bt.with_pipeline_layout(vk_dev, layout)?)
		})
		.transpose()?;*/
		let transparency_pipeline = if let Some(bt) = builder_transparency {
			let built_transparency_pipeline =
				bt.with_auto_layout(vk_dev.clone(), |sets| pipeline_sampler_setup(sets, &samplers))?;
			print_pipeline_descriptors_info(built_transparency_pipeline.as_ref());
			Some(built_transparency_pipeline)
		} else {
			None
		};

		Ok(Pipeline {
			descriptor_set_allocator,
			pipeline,
			transparency_pipeline,
		})
	}

	/// Create a pipeline from a YAML pipeline configuration file.
	pub fn new_from_yaml(
		yaml_filename: &str,
		subpass: Subpass,
		transparency_subpass: Option<Subpass>,
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

		let mut color_blend_state = ColorBlendState::new(subpass.num_color_attachments());
		if deserialized.alpha_blending {
			color_blend_state = color_blend_state.blend_alpha();
		}

		let fs_info = deserialized.fragment_shader.map(|fs| (fs, color_blend_state));
		let fs_transparency_info = deserialized
			.fragment_shader_transparency
			.map(|fst| (fst, transparency_subpass.unwrap()));

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
			subpass,
			depth_op,
			true,
			descriptor_set_allocator,
		)
	}

	pub fn bind<L>(&self, command_buffer: &mut AutoCommandBufferBuilder<L>)
	{
		command_buffer.bind_pipeline_graphics(self.pipeline.clone());
	}
	pub fn bind_transparency<L>(&self, command_buffer: &mut AutoCommandBufferBuilder<L>) -> Result<(), TransparencyNotEnabled>
	{
		command_buffer.bind_pipeline_graphics(self.transparency_pipeline.clone().ok_or(TransparencyNotEnabled)?);
		Ok(())
	}

	pub fn layout(&self) -> Arc<PipelineLayout>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self.pipeline.as_ref();
		pipeline_ref.layout().clone()
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
			.ok_or("Pipeline::new_descriptor_set: invalid descriptor set index")?
			.clone();
		Ok(PersistentDescriptorSet::new(
			&self.descriptor_set_allocator,
			set_layout,
			writes,
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

fn get_entry_point<'a>(
	shader_module: &'a Arc<ShaderModule>,
	entry_point_name: &str,
) -> Result<EntryPoint<'a>, GenericEngineError>
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
		.input_interface()
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
		ShaderScalarType::Float => [
			Format::R32_SFLOAT,
			Format::R32G32_SFLOAT,
			Format::R32G32B32_SFLOAT,
			Format::R32G32B32A32_SFLOAT,
		],
		ShaderScalarType::Sint => [
			Format::R32_SINT,
			Format::R32G32_SINT,
			Format::R32G32B32_SINT,
			Format::R32G32B32A32_SINT,
		],
		ShaderScalarType::Uint => [
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
