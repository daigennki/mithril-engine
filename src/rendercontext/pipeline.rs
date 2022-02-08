/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::io::Read;
use vulkano::pipeline::graphics::viewport::*;
use vulkano::shader::ShaderModule;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::RenderPass;
use vulkano::render_pass::Subpass;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::vertex_input::VertexInputRate;
use vulkano::pipeline::graphics::vertex_input::VertexInputBindingDescription;
use vulkano::pipeline::graphics::vertex_input::VertexInputAttributeDescription;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::input_assembly::PrimitiveTopology;
use vulkano::pipeline::PartialStateMode;
use vulkano::pipeline::StateMode;
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::color_blend::AttachmentBlend;
use vulkano::pipeline::graphics::color_blend::BlendOp;
use vulkano::pipeline::graphics::color_blend::BlendFactor;
use vulkano::pipeline::PipelineLayout;
use vulkano::format::Format;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::sampler::Sampler;
use vulkano::shader::DescriptorRequirements;
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
		let input_assembly_state = InputAssemblyState{
			topology: PartialStateMode::Fixed(primitive_topology),
			primitive_restart_enable: StateMode::Fixed(false)
		};
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

		let pipeline_built = build_pipeline_common(
			vk_dev.clone(), input_assembly_state, vertex_input_state, width, height,
			vs.clone(), fs.clone(), 
			subpass,
			&samplers
		)?;
			
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
			&self.samplers
		)?;

		Ok(())
	}

	pub fn bind(&self, command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, StandardCommandPoolBuilder>) 
	{
		command_buffer.bind_pipeline_graphics(self.pipeline.clone());
	}

	pub fn layout(&self) -> Arc<PipelineLayout>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self.pipeline.as_ref();
		pipeline_ref.layout().clone()
	}

	pub fn get_descriptor_requirements(&self) -> impl ExactSizeIterator<Item = ((u32, u32), &DescriptorRequirements)>
	{
		self.pipeline.descriptor_requirements()
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

fn get_format_components(format: Format) -> Result<u32, Box<dyn std::error::Error>> 
{
	match format {
		Format::R32_SFLOAT => Ok(1),
		Format::R32G32_SFLOAT => Ok(2),
		Format::R32G32B32_SFLOAT => Ok(3),
		Format::R32G32B32A32_SFLOAT => Ok(4),
		_ => Err("unsupported vertex input format".into())
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
	samplers: &Vec<(usize, u32, Arc<Sampler>)>
) -> Result<Arc<GraphicsPipeline>, Box<dyn std::error::Error>>
{
	let viewport = Viewport{ 
		origin: [ 0.0, 0.0 ],
		dimensions: [ width as f32, height as f32 ],
		depth_range: (0.0..1.0)
	};

	// TODO: add parameters to change this
	let color_blend_state = ColorBlendState::new(subpass.num_color_attachments()).blend(
		AttachmentBlend {
			color_op: BlendOp::Add,
			color_source: BlendFactor::One,
			color_destination: BlendFactor::OneMinusSrcAlpha,
			alpha_op: BlendOp::Add,
			alpha_source: BlendFactor::One,
			alpha_destination: BlendFactor::OneMinusSrcAlpha,
		},
	);
	
	// do some building
	let mut pipeline_builder = GraphicsPipeline::start()
		.input_assembly_state(input_assembly_state)
		.vertex_input_state(vertex_input_state)
		.viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
		.color_blend_state(color_blend_state)
		.render_pass(subpass);
	
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
		let mut sets_vec: Vec<_> = sets.into();
		for (set_i, binding_i, sampler) in samplers {
			match sets_vec.get_mut(*set_i) {
				Some(s) => {
					s.set_immutable_samplers(*binding_i, [ sampler.clone() ]);
				}
				None => {
					log::warn!("Set index {} for sampler is out of bounds, ignoring!", set_i);
				}
			}
		}
	})?;

	Ok(pipeline)
}
