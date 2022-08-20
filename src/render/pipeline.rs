/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::path::Path;
use std::fs::File;
use vulkano::shader::ShaderModule;
use vulkano::render_pass::{ RenderPass, Subpass };
use vulkano::pipeline::{ GraphicsPipeline, PipelineLayout };
use vulkano::pipeline::graphics::viewport::*;
use vulkano::pipeline::graphics::vertex_input::{ VertexInputState, VertexInputRate, VertexInputBindingDescription };
use vulkano::pipeline::graphics::vertex_input::VertexInputAttributeDescription;
use vulkano::pipeline::graphics::input_assembly::{ InputAssemblyState, PrimitiveTopology };
use vulkano::pipeline::graphics::color_blend::{ ColorBlendState, AttachmentBlend };
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::format::Format;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::sampler::{ SamplerCreateInfo, Sampler, Filter };
use vulkano::descriptor_set::{ 
	layout::DescriptorType, WriteDescriptorSet, PersistentDescriptorSet,
};
use vulkano::device::DeviceOwned;
use spirv_reflect::types::image::ReflectFormat;
use serde::{Serialize, Deserialize};

use crate::GenericEngineError;


pub struct Pipeline
{
	vs: Arc<ShaderModule>,
	fs: Option<Arc<ShaderModule>>,
	samplers: Vec<(usize, u32, Arc<Sampler>)>,
	pipeline: Arc<GraphicsPipeline>,
	subpass: Subpass
}
impl Pipeline
{
	pub fn new( 
		primitive_topology: PrimitiveTopology,	
		vs_filename: String,
		fs_filename: Option<String>,
		samplers: Vec<(usize, u32, Arc<Sampler>)>,	// set: usize, binding: u32, sampler: Arc<Sampler>
		render_pass: Arc<RenderPass>, 
		width: u32, height: u32,
	) -> Result<Self, GenericEngineError>
	{
		let vk_dev = render_pass.device().clone();

		// load vertex shader
		log::info!("Loading vertex shader {}...", vs_filename);
		let (vs, vertex_input_state) = load_spirv_vertex(vk_dev.clone(), &Path::new("shaders").join(vs_filename))?;

		// load fragment shader (optional)
		let fs = fs_filename.map(|f| {
			log::info!("Loading fragment shader {}...", f);
			load_spirv(vk_dev.clone(), &Path::new("shaders").join(f))
		}).transpose()?;

		let subpass = Subpass::from(render_pass.clone(), 0).ok_or("Subpass 0 for render pass doesn't exist!")?;
		let input_assembly_state = InputAssemblyState::new().topology(primitive_topology);
		let color_blend_state = color_blend_state_from_subpass(&subpass);

		let pipeline_built = build_pipeline_common(
			vk_dev.clone(), input_assembly_state, 
			vertex_input_state, 
			width, height,
			vs.clone(), fs.clone(), 
			subpass.clone(),
			&samplers,
			color_blend_state
		)?;

		log::debug!("Built pipeline with descriptors:");
		for ((set, binding), req) in pipeline_built.descriptor_requirements() {
			log::debug!(
				"set {}, binding {}: {}x {}", 
				set, binding, req.descriptor_count, 
				&print_descriptor_types(&req.descriptor_types)
			);
		}
			
		Ok(Pipeline{
			vs: vs,
			fs: fs,
			samplers: samplers,
			pipeline: pipeline_built,
			subpass: subpass,
		})
	}

	/// Create a pipeline from a YAML pipeline configuration file.
	pub fn new_from_yaml(yaml_filename: &str, render_pass: Arc<RenderPass>, width: u32, height: u32)
		-> Result<Self, GenericEngineError>
	{
		log::info!("Loading pipeline definition file '{}'...", yaml_filename);

		let yaml_reader = File::open(Path::new("shaders").join(yaml_filename))?;
		let deserialized: PipelineConfig = serde_yaml::from_reader(yaml_reader)?;

		let mut generated_samplers = Vec::<(usize, u32, Arc<Sampler>)>::new();
		if let Some(sampler_configs) = deserialized.samplers {
			for sampler_config in sampler_configs {
				let mut sampler_create_info = SamplerCreateInfo::default();
				if let Some(f) = sampler_config.mag_filter {
					sampler_create_info.mag_filter = filter_str_to_enum(&f)?;
				}
				if let Some(f) = sampler_config.min_filter {
					sampler_create_info.min_filter = filter_str_to_enum(&f)?;
				}

				let new_sampler = Sampler::new(render_pass.device().clone(), sampler_create_info)?;
				generated_samplers.push((sampler_config.set, sampler_config.binding, new_sampler));
			}
		}

		Pipeline::new(
			deserialized.primitive_topology, 
			deserialized.vertex_shader, 
			deserialized.fragment_shader, 
			generated_samplers, render_pass, width, height
		)
	}

	pub fn resize_viewport(&mut self, width: u32, height: u32) -> Result<(), GenericEngineError>
	{
		self.pipeline = build_pipeline_common(
			self.pipeline.device().clone(), 
			self.pipeline.input_assembly_state().clone(),
			self.pipeline.vertex_input_state().clone(), width, height,
			self.vs.clone(), self.fs.clone(), 
			self.subpass.clone(),
			&self.samplers,
			self.pipeline.color_blend_state().cloned()
		)?;

		Ok(())
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
	pub fn new_descriptor_set(&self, set_number: usize, writes: impl IntoIterator<Item = WriteDescriptorSet>)
		-> Result<Arc<PersistentDescriptorSet>, GenericEngineError>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self.pipeline.as_ref();
		let set_layout = pipeline_ref.layout().set_layouts().get(set_number)
			.ok_or("Pipeline::new_descriptor_set: invalid descriptor set index")?
			.clone();
		Ok(PersistentDescriptorSet::new(set_layout, writes)?)
	}
}

#[derive(Serialize, Deserialize)]
struct PipelineSamplerConfig {
	set: usize,
	binding: u32,
	min_filter: Option<String>,
	mag_filter: Option<String>
}
#[derive(Serialize, Deserialize)]
struct PipelineConfig {
	vertex_shader: String,
	fragment_shader: Option<String>,

	#[serde(with = "PrimitiveTopologyDef")]
	primitive_topology: PrimitiveTopology,

	samplers: Option<Vec<PipelineSamplerConfig>>
}

// copy of `vulkano::pipeline::graphics::input_assembly::PrimitiveTopology` so we can more directly (de)serialize it
#[derive(Serialize, Deserialize)]
#[serde(remote = "PrimitiveTopology")]
enum PrimitiveTopologyDef {
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
		_ => return Err("Invalid sampler filter".into())
	})
}

fn load_spirv(device: Arc<vulkano::device::Device>, path: &Path) 
	-> Result<Arc<vulkano::shader::ShaderModule>, GenericEngineError>
{
	let spv_data = std::fs::read(path)?;
	Ok(unsafe { vulkano::shader::ShaderModule::from_bytes(device, &spv_data) }?)
}

/// Load the SPIR-V file, and also automatically determine the given vertex shader's vertex inputs using information from the SPIR-V file.
fn load_spirv_vertex(device: Arc<vulkano::device::Device>, path: &Path)
	-> Result<(Arc<vulkano::shader::ShaderModule>, VertexInputState), GenericEngineError>
{
	let spv_data = std::fs::read(path)?;
	let shader_module = spirv_reflect::ShaderModule::load_u8_data(&spv_data)?;
	let input_variables = shader_module.enumerate_input_variables(Some("main"))?;

	let mut i: u32 = 0;
	let mut vertex_input_state = VertexInputState::new();
	for input_var in &input_variables {
		let vertex_format = reflect_format_to_vulkano_format(input_var.format)?;
		let stride = vertex_format.components().iter().fold(0, |acc, c| acc + (*c as u32)) / 8;

		vertex_input_state = vertex_input_state
			.binding(i, VertexInputBindingDescription{ stride: stride, input_rate: VertexInputRate::Vertex })
			.attribute(i, VertexInputAttributeDescription{ binding: i, format: vertex_format, offset: 0 });
		i += 1;
	}

	Ok((unsafe { vulkano::shader::ShaderModule::from_bytes(device, &spv_data) }?, vertex_input_state))
}

#[derive(Debug)]
pub struct UnsupportedVertexInputFormat;
impl std::error::Error for UnsupportedVertexInputFormat {}
impl std::fmt::Display for UnsupportedVertexInputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "unsupported vertex input format")
    }
}
fn reflect_format_to_vulkano_format(reflect_format: spirv_reflect::types::image::ReflectFormat) 
	-> Result<Format, UnsupportedVertexInputFormat>
{
	Ok(match reflect_format {
		ReflectFormat::R32_UINT => Format::R32_UINT,
		ReflectFormat::R32_SINT => Format::R32_UINT,
		ReflectFormat::R32_SFLOAT => Format::R32_SFLOAT,
		ReflectFormat::R32G32_UINT => Format::R32G32_UINT,
		ReflectFormat::R32G32_SINT => Format::R32G32_SINT,
		ReflectFormat::R32G32_SFLOAT => Format::R32G32_SFLOAT,
		ReflectFormat::R32G32B32_UINT => Format::R32G32B32_UINT,
		ReflectFormat::R32G32B32_SINT => Format::R32G32B32_SINT,
		ReflectFormat::R32G32B32_SFLOAT => Format::R32G32B32_SFLOAT,
		ReflectFormat::R32G32B32A32_UINT => Format::R32G32B32A32_UINT,
		ReflectFormat::R32G32B32A32_SINT => Format::R32G32B32A32_SINT,
		ReflectFormat::R32G32B32A32_SFLOAT => Format::R32G32B32A32_SFLOAT,
		_ => return Err(UnsupportedVertexInputFormat)
	})
}

fn color_blend_state_from_subpass(subpass: &Subpass) -> Option<ColorBlendState>
{
	// only enable blending for the first attachment.
	// This blending configuration is for textures that are *not* premultiplied by alpha.
	// TODO: expose an option for premultiplied alpha. also be careful about using plain RGBA colors instead of textures, as
	// those have to be premultiplied too.
	if subpass.num_color_attachments() > 0 {
		let mut blend_state = ColorBlendState::new(subpass.num_color_attachments());
		blend_state.attachments[0].blend = Some(AttachmentBlend::alpha());
		Some(blend_state)
	} else {
		None
	}
}

fn build_pipeline_common(
	vk_dev: Arc<vulkano::device::Device>, 
	input_assembly_state: InputAssemblyState,
	vertex_input_state: VertexInputState,
	width: u32, height: u32,
	vs: Arc<ShaderModule>,
	fs: Option<Arc<ShaderModule>>,
	subpass: Subpass,
	samplers: &Vec<(usize, u32, Arc<Sampler>)>,
	color_blend_state: Option<ColorBlendState>
) -> Result<Arc<GraphicsPipeline>, GenericEngineError>
{
	let viewport = Viewport{ 
		origin: [ 0.0, 0.0 ],
		dimensions: [ width as f32, height as f32 ],
		depth_range: (0.0..1.0)
	};
	
	// do some building
	let mut pipeline_builder = GraphicsPipeline::start()
		.input_assembly_state(input_assembly_state)
		.vertex_input_state(vertex_input_state)
		.depth_stencil_state(DepthStencilState::simple_depth_test())
		.viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
		.render_pass(subpass);
	
	if let Some(c) = color_blend_state {
		pipeline_builder = pipeline_builder.color_blend_state(c);
	}
	
	let vs_entry = vs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
	pipeline_builder = pipeline_builder.vertex_shader(vs_entry, ());

	if let Some(fs_exists) = fs.as_ref() {
		let fs_entry = fs_exists.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
		pipeline_builder = pipeline_builder.fragment_shader(fs_entry, ());
	}

	// build pipeline with immutable samplers, if it needs any
	let pipeline = pipeline_builder.with_auto_layout(vk_dev, |sets| {
		for (set_i, binding_i, sampler) in samplers {
			match sets.get_mut(*set_i) {
				Some(s) => match s.bindings.get_mut(binding_i) {
					Some(b) => b.immutable_samplers = vec![ sampler.clone() ],
					None => log::warn!("Binding {} doesn't exist in set {}, ignoring!", binding_i, set_i)
				},
				None => log::warn!("Set index {} for sampler is out of bounds, ignoring!", set_i)
			}
		}
	})?;

	Ok(pipeline)
}

fn print_descriptor_types(types: &Vec<DescriptorType>) -> String
{
	let mut out_str = String::new();
	let mut first = true;
	for ty in types {
		if first {
			first = false;
		} else {
			out_str += "/";
		}
		out_str += match ty {
			DescriptorType::Sampler => "Sampler",
			DescriptorType::CombinedImageSampler => "Combined image sampler",
			DescriptorType::SampledImage => "Sampled image",
			DescriptorType::StorageImage => "Storage image",
			DescriptorType::UniformTexelBuffer => "Uniform texel buffer",
			DescriptorType::StorageTexelBuffer => "Storage texel buffer",
			DescriptorType::UniformBuffer => "Uniform buffer",
			DescriptorType::StorageBuffer => "Storage buffer",
			DescriptorType::UniformBufferDynamic => "Dynamic uniform buffer",
			DescriptorType::StorageBufferDynamic => "Dynamic storage buffer",
			DescriptorType::InputAttachment => "Input attachment",
			_ => "(unknown)"
		}
	}
	out_str
}

