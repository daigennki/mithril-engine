/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::io::Read;
use vulkano::shader::ShaderModule;
use vulkano::render_pass::{ RenderPass, Subpass };
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::PipelineLayout;
use vulkano::pipeline::graphics::viewport::*;
use vulkano::pipeline::graphics::vertex_input::{ VertexInputState, VertexInputRate, VertexInputBindingDescription };
use vulkano::pipeline::graphics::vertex_input::VertexInputAttributeDescription;
use vulkano::pipeline::graphics::input_assembly::{ InputAssemblyState, PrimitiveTopology };
use vulkano::pipeline::graphics::color_blend::{ ColorBlendState, AttachmentBlend, BlendOp, BlendFactor };
use vulkano::format::Format;
use vulkano::command_buffer::{ AutoCommandBufferBuilder, PrimaryAutoCommandBuffer };
use vulkano::sampler::Sampler;
use vulkano::descriptor_set::{ layout::DescriptorType, WriteDescriptorSet, PersistentDescriptorSet };
use std::mem::size_of;

pub struct Pipeline
{
	vs: Arc<ShaderModule>,
	fs: Option<Arc<ShaderModule>>,
	samplers: Vec<(usize, u32, Arc<Sampler>)>,
	pipeline: Arc<GraphicsPipeline>
}
impl Pipeline
{
	pub fn new(
		vk_dev: Arc<vulkano::device::Device>, 
		primitive_topology: PrimitiveTopology,
		vertex_input: impl IntoIterator<Item = Format>,
		vs_filename: String, 
		fs_filename: Option<String>,
		samplers: Vec<(usize, u32, Arc<Sampler>)>,	// set: usize, binding: u32, sampler: Arc<Sampler>
		render_pass: Arc<RenderPass>, 
		width: u32, height: u32,
	) -> Result<Pipeline, Box<dyn std::error::Error>>
	{
		// load vertex shader
		log::info!("Loading vertex shader {}...", &vs_filename);
		let vs = load_spirv(vk_dev.clone(), &format!("shaders/{}", &vs_filename))?;

		// load fragment shader (optional)
		let fs = match fs_filename {
			Some(f) => {
				log::info!("Loading fragment shader {}...", f);
				Some(load_spirv(vk_dev.clone(), &format!("shaders/{}", f))?)
			}
			None => None
		};

		let subpass = Subpass::from(render_pass.clone(), 0).ok_or("Subpass 0 for render pass doesn't exist!")?;
		let input_assembly_state = InputAssemblyState::new().topology(primitive_topology);
		let vertex_input_state = vertex_input_state_from_formats(vertex_input)?;
		let color_blend_state = color_blend_state_from_subpass(&subpass);

		let pipeline_built = build_pipeline_common(
			vk_dev.clone(), input_assembly_state, 
			vertex_input_state, 
			width, height,
			vs.clone(), fs.clone(), 
			subpass,
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
			pipeline: pipeline_built
		})
	}

	pub fn resize_viewport(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>>
	{
		self.pipeline = build_pipeline_common(
			self.pipeline.device().clone(), 
			self.pipeline.input_assembly_state().clone(),
			self.pipeline.vertex_input_state().clone(), width, height,
			self.vs.clone(), self.fs.clone(), 
			self.pipeline.subpass().clone(),
			&self.samplers,
			self.pipeline.color_blend_state().cloned()
		)?;

		Ok(())
	}

	pub fn bind(&self, command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) 
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
	pub fn new_descriptor_set(&self, set: usize, writes: impl IntoIterator<Item = WriteDescriptorSet>)
		-> Result<Arc<PersistentDescriptorSet>, Box<dyn std::error::Error>>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self.pipeline.as_ref();
		let set_layout = pipeline_ref.layout().set_layouts().get(set)
			.ok_or("Pipeline::new_descriptor_set: invalid descriptor set index")?
			.clone();
		Ok(PersistentDescriptorSet::new(set_layout, writes)?)
	}
}

fn load_spirv(device: Arc<vulkano::device::Device>, filename: &str) 
	-> Result<Arc<vulkano::shader::ShaderModule>, Box<dyn std::error::Error>>
{
	let mut spv_file = std::fs::File::open(filename)
		.or_else(|e| Err(format!("Failed to open '{}': {}", filename, e)))?;

	let mut spv_data: Vec<u8> = Vec::new();
	spv_file.read_to_end(&mut spv_data)
		.or_else(|e| Err(format!("Failed to read '{}': {}", filename, e)))?;

	Ok(unsafe { vulkano::shader::ShaderModule::from_bytes(device, &spv_data) }?)
}

#[derive(Debug)]
pub struct UnsupportedVertexInputFormat;
impl std::error::Error for UnsupportedVertexInputFormat {}
impl std::fmt::Display for UnsupportedVertexInputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "unsupported vertex input format")
    }
}
fn get_format_components(format: Format) -> Result<u32, UnsupportedVertexInputFormat> 
{
	match format {
		Format::R32_SFLOAT => Ok(1),
		Format::R32G32_SFLOAT => Ok(2),
		Format::R32G32B32_SFLOAT => Ok(3),
		Format::R32G32B32A32_SFLOAT => Ok(4),
		_ => Err(UnsupportedVertexInputFormat)
	}
}
fn vertex_input_state_from_formats(vertex_input: impl IntoIterator<Item = Format>) 
	-> Result<VertexInputState, UnsupportedVertexInputFormat>
{
	// Automatically generate vertex input state from input formats and strides.
	// This assumes bindings in order starting at 0, tightly packed, and with 0 offset - the typical use case.
	// VertexInputState will remain as default if vertex_input was empty.
	let mut i: u32 = 0;
	let mut vertex_input_state = VertexInputState::new();
	for vertex_format in vertex_input {
		let stride = get_format_components(vertex_format)? * size_of::<f32>() as u32;

		vertex_input_state = vertex_input_state
			.binding(i, VertexInputBindingDescription{ stride: stride, input_rate: VertexInputRate::Vertex })
			.attribute(i, VertexInputAttributeDescription{ binding: i, format: vertex_format, offset: 0 });
		i += 1;
	}
	Ok(vertex_input_state)
}

fn color_blend_state_from_subpass(subpass: &Subpass) -> Option<ColorBlendState>
{
	// only enable blending for the first attachment.
	// this blending configuration will require textures to be premultiplied by alpha.
	// TODO: what do we do about PNG images, which are *not* premultiplied by alpha, according to the PNG spec?
	if subpass.num_color_attachments() > 0 {
		let mut new_color_blend_state = ColorBlendState::new(subpass.num_color_attachments());
		match new_color_blend_state.attachments.get_mut(0) {
			Some(a) => a.blend = Some(AttachmentBlend{
				color_op: BlendOp::Add,
				color_source: BlendFactor::One,
				color_destination: BlendFactor::OneMinusSrcAlpha,
				alpha_op: BlendOp::Add,
				alpha_source: BlendFactor::One,
				alpha_destination: BlendFactor::OneMinusSrcAlpha
			}),
			None => ()
		}
		Some(new_color_blend_state)
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
) -> Result<Arc<GraphicsPipeline>, Box<dyn std::error::Error>>
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
		.viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
		.render_pass(subpass);

	match color_blend_state {
		Some(c) => pipeline_builder = pipeline_builder.color_blend_state(c),
		None => ()
	}
	
	let vs_entry = vs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
	pipeline_builder = pipeline_builder.vertex_shader(vs_entry, ());

	let fs_moved;
	match fs {
		Some(fs_exists) => {
			fs_moved = fs_exists;
			let fs_entry = fs_moved.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
			pipeline_builder = pipeline_builder.fragment_shader(fs_entry, ());
		}
		None => ()
	}

	// build pipeline with immutable samplers, if it needs any
	let pipeline = pipeline_builder.with_auto_layout(vk_dev, |sets| {
		for (set_i, binding_i, sampler) in samplers {
			match sets.get_mut(*set_i) {
				Some(s) => {
					match s.bindings.get_mut(binding_i) {
						Some(b) => b.immutable_samplers = vec![ sampler.clone() ],
						None => log::warn!("Binding {} doesn't exist in set {}, ignoring!", binding_i, set_i)
					}
				}
				None => {
					log::warn!("Set index {} for sampler is out of bounds, ignoring!", set_i);
				}
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
