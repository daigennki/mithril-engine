/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::{layout::DescriptorSetLayoutCreateInfo, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, ColorBlendState},
	depth_stencil::{CompareOp, DepthState, DepthStencilState},
	input_assembly::{InputAssemblyState, PrimitiveTopology},
	rasterization::{CullMode, FrontFace, RasterizationState},
	vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState},
	viewport::ViewportState,
};
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, StateMode};
use vulkano::render_pass::Subpass;
use vulkano::sampler::{Filter, Sampler, SamplerCreateInfo};
use vulkano::shader::{EntryPoint, ShaderInterfaceEntryType, ShaderModule, ShaderScalarType};

use crate::GenericEngineError;

pub struct Pipeline
{
	pipeline: Arc<GraphicsPipeline>,
}
impl Pipeline
{
	pub fn new(
		primitive_topology: PrimitiveTopology,
		vs_filename: String,
		fs_filename: Option<String>,
		samplers: Vec<(usize, u32, Arc<Sampler>)>, // set: usize, binding: u32, sampler: Arc<Sampler>
		subpass: Subpass,
		depth_op: CompareOp,
		color_blend_state: Option<ColorBlendState>,
		depth_write: bool,
		reuse_layout: Option<Arc<PipelineLayout>>,
	) -> Result<Self, GenericEngineError>
	{
		let vk_dev = subpass.render_pass().device().clone();

		let (vs, vertex_input_state) = load_spirv_vertex(vk_dev.clone(), &Path::new("shaders").join(vs_filename))?;

		let mut input_assembly_state = InputAssemblyState::new().topology(primitive_topology);
		if primitive_topology == PrimitiveTopology::TriangleStrip {
			input_assembly_state = input_assembly_state.primitive_restart_enable();
		}

		let depth_stencil_state = DepthStencilState {
			depth: Some(DepthState {
				enable_dynamic: false,
				write_enable: StateMode::Fixed(depth_write),
				compare_op: StateMode::Fixed(depth_op),
			}),
			..Default::default()
		};

		let rasterization_state = RasterizationState::new()
			.cull_mode(CullMode::Back)
			.front_face(FrontFace::CounterClockwise);

		// do some building
		let mut pipeline_builder = GraphicsPipeline::start()
			.vertex_shader(get_entry_point(&vs, "main")?, ())
			.vertex_input_state(vertex_input_state)
			.input_assembly_state(input_assembly_state)
			.viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
			.rasterization_state(rasterization_state)
			.depth_stencil_state(depth_stencil_state)
			.render_pass(subpass);

		if let Some(c) = color_blend_state {
			pipeline_builder = pipeline_builder.color_blend_state(c);
		}

		// load fragment shader (optional)
		let fs;
		if let Some(f) = fs_filename {
			fs = load_spirv(vk_dev.clone(), &Path::new("shaders").join(f))?;
			pipeline_builder = pipeline_builder.fragment_shader(get_entry_point(&fs, "main")?, ());
		}

		let pipeline_built = match reuse_layout {
			Some(layout) => pipeline_builder.with_pipeline_layout(vk_dev, layout)?,
			None => {
				// build pipeline with immutable samplers, if it needs any
				pipeline_builder.with_auto_layout(vk_dev, |sets| pipeline_sampler_setup(sets, &samplers))?
			}
		};
		print_pipeline_descriptors_info(&pipeline_built);

		Ok(Pipeline { pipeline: pipeline_built })
	}

	/// Create a pipeline from a YAML pipeline configuration file.
	pub fn new_from_yaml(yaml_filename: &str, subpass: Subpass) -> Result<Self, GenericEngineError>
	{
		log::info!("Loading pipeline definition file '{}'...", yaml_filename);

		let yaml_reader = File::open(Path::new("shaders").join(yaml_filename))?;
		let deserialized: PipelineConfig = serde_yaml::from_reader(yaml_reader)?;
		let vk_dev = subpass.render_pass().device().clone();
		let generated_samplers = deserialized
			.samplers
			.iter()
			.map(|sampler_config| {
				let new_sampler = Sampler::new(vk_dev.clone(), sampler_config.as_create_info()?)?;
				Ok((sampler_config.set, sampler_config.binding, new_sampler))
			})
			.collect::<Result<_, GenericEngineError>>()?;
		let color_blend_state = color_blend_state_from_subpass(&subpass);

		Pipeline::new(
			deserialized.primitive_topology,
			deserialized.vertex_shader,
			deserialized.fragment_shader,
			generated_samplers,
			subpass,
			CompareOp::Less,
			color_blend_state,
			true,
			None
		)
	}

	pub fn bind<L>(&self, command_buffer: &mut AutoCommandBufferBuilder<L>)
	{
		command_buffer.bind_pipeline_graphics(self.pipeline.clone());
	}

	pub fn layout(&self) -> Arc<PipelineLayout>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self.pipeline.as_ref();
		pipeline_ref.layout().clone()
	}

	/// Create a new persistent descriptor set for use with the descriptor set slot at `set_number`, writing `writes`
	/// into the descriptor set.
	pub fn new_descriptor_set(
		&self, set_number: usize, writes: impl IntoIterator<Item = WriteDescriptorSet>,
	) -> Result<Arc<PersistentDescriptorSet>, GenericEngineError>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self.pipeline.as_ref();
		let set_layout = pipeline_ref
			.layout()
			.set_layouts()
			.get(set_number)
			.ok_or("Pipeline::new_descriptor_set: invalid descriptor set index")?
			.clone();
		Ok(PersistentDescriptorSet::new(set_layout, writes)?)
	}
}

#[derive(Deserialize)]
struct PipelineSamplerConfig
{
	set: usize,
	binding: u32,
	min_filter: Option<String>,
	mag_filter: Option<String>,
}
impl PipelineSamplerConfig
{
	fn as_create_info(&self) -> Result<SamplerCreateInfo, GenericEngineError>
	{
		let mut sampler_create_info = SamplerCreateInfo::default();
		if let Some(f) = self.mag_filter.as_ref() {
			sampler_create_info.mag_filter = filter_str_to_enum(f)?;
		}
		if let Some(f) = self.min_filter.as_ref() {
			sampler_create_info.min_filter = filter_str_to_enum(f)?;
		}
		Ok(sampler_create_info)
	}
}

/*#[derive(Deserialize)]
struct PipelineBlendState
{
	
}*/

#[derive(Deserialize)]
struct PipelineConfig
{
	vertex_shader: String,
	fragment_shader: Option<String>,

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

fn filter_str_to_enum(filter_str: &str) -> Result<Filter, GenericEngineError>
{
	Ok(match filter_str {
		"Nearest" => Filter::Nearest,
		"Linear" => Filter::Linear,
		_ => return Err("Invalid sampler filter".into()),
	})
}

fn get_entry_point<'a>(
	shader_module: &'a Arc<ShaderModule>, entry_point_name: &str,
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
	device: Arc<vulkano::device::Device>, path: &Path,
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
				.binding(binding, VertexInputBindingDescription { stride, input_rate: VertexInputRate::Vertex })
				.attribute(binding, VertexInputAttributeDescription { binding, format, offset: 0 })
		});

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

fn color_blend_state_from_subpass(subpass: &Subpass) -> Option<ColorBlendState>
{
	// Only enable blending for the first attachment.
	// This blending configuration is for textures that are *not* premultiplied by alpha.
	// TODO: Expose an option for premultiplied alpha. Also be careful about using plain RGBA colors instead of textures, as
	// those have to be premultiplied too.
	if subpass.num_color_attachments() > 0 {
		let mut blend_state = ColorBlendState::new(subpass.num_color_attachments());
		blend_state.attachments[0].blend = Some(AttachmentBlend::alpha());
		Some(blend_state)
	} else {
		None
	}
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

fn print_pipeline_descriptors_info(pipeline: &GraphicsPipeline)
{
	log::debug!("Built pipeline with descriptors:");
	for ((set, binding), req) in pipeline.descriptor_requirements() {
		let desc_count_string = match req.descriptor_count {
			Some(count) => format!("{}x", count),
			None => "runtime-sized array of".into(),
		};

		log::debug!("set {}, binding {}: {} {:?}", set, binding, desc_count_string, req.descriptor_types);
	}
}
