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
use vulkano::pipeline::graphics::color_blend::{ ColorBlendState, AttachmentBlend };
use vulkano::format::Format;
use vulkano::command_buffer::{ AutoCommandBufferBuilder, PrimaryAutoCommandBuffer };
use vulkano::sampler::Sampler;
use vulkano::descriptor_set::{ layout::DescriptorType, WriteDescriptorSet, PersistentDescriptorSet };
use vulkano::device::DeviceOwned;
use spirv_reflect::types::image::ReflectFormat;
use yaml_rust::YamlLoader;

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
		vs_filename: &str,
		fs_filename: Option<&str>,
		samplers: Vec<(usize, u32, Arc<Sampler>)>,	// set: usize, binding: u32, sampler: Arc<Sampler>
		render_pass: Arc<RenderPass>, 
		width: u32, height: u32,
	) -> Result<Pipeline, Box<dyn std::error::Error>>
	{
		let vk_dev = render_pass.device().clone();

		// load vertex shader
		log::info!("Loading vertex shader {}...", vs_filename);
		let (vs, vertex_input_state) = load_spirv_vertex(vk_dev.clone(), &format!("shaders/{}", vs_filename))?;

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
			subpass: subpass
		})
	}

	/// Create a pipeline from a YAML pipeline configuration file.
	pub fn new_from_yaml(yaml_filename: &str, render_pass: Arc<RenderPass>, width: u32, height: u32)
		-> Result<Pipeline, Box<dyn std::error::Error>>
	{
		log::info!("Loading pipeline definition file '{}'...", yaml_filename);

		let mut yaml_file = std::fs::File::open(format!("shaders/{}", yaml_filename))?;
		let mut yaml_string = String::new();
		yaml_file.read_to_string(&mut yaml_string)?;

		let yaml_docs = YamlLoader::load_from_str(&yaml_string)?;
		let yaml_doc = &yaml_docs[0];

		let primitive_topology_str = yaml_doc["PrimitiveTopology"].as_str().ok_or("Primitive topology not specified!")?;
		let primitive_topology = prim_topo_str_to_enum(primitive_topology_str)?;

		let vs_filename = yaml_doc["VertexShader"].as_str().ok_or("Vertex shader not specified!")?;
		let fs_filename = yaml_doc["FragmentShader"].as_str();

		let mut generated_samplers: Vec<(usize, u32, Arc<Sampler>)> = vec![];
		match yaml_doc["Samplers"].as_hash() {
			Some(samplers) => {
				for (set_key, set) in samplers {
					let set_number = set_key.as_i64().ok_or("Invalid set number in sampler list")?;
					for (binding_key, binding) in set.as_hash().ok_or("Invalid sampler binding")? {
						let binding_number = binding_key.as_i64().ok_or("Invalid binding number in sampler list")?;
						let sampler = sampler_from_yaml(binding, render_pass.clone())?;

						generated_samplers.push((set_number.try_into()?, binding_number.try_into()?, sampler));

						log::debug!("created sampler at set {}, binding {}", set_number, binding_number);
					}
				}
			}
			None => ()
		}

		Pipeline::new(primitive_topology, vs_filename, fs_filename, generated_samplers, render_pass, width, height)
	}

	pub fn resize_viewport(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>>
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
	pub fn new_descriptor_set(&self, set_number: usize, writes: impl IntoIterator<Item = WriteDescriptorSet>)
		-> Result<Arc<PersistentDescriptorSet>, Box<dyn std::error::Error>>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self.pipeline.as_ref();
		let set_layout = pipeline_ref.layout().set_layouts().get(set_number)
			.ok_or("Pipeline::new_descriptor_set: invalid descriptor set index")?
			.clone();
		Ok(PersistentDescriptorSet::new(set_layout, writes)?)
	}
}

fn sampler_from_yaml(sampler_yaml: &yaml_rust::Yaml, render_pass: Arc<RenderPass>) -> Result<Arc<Sampler>, Box<dyn std::error::Error>>
{
	let mut sampler_create_info = vulkano::sampler::SamplerCreateInfo::default();

	for (item_name, data) in sampler_yaml.as_hash().ok_or("Invalid sampler entry")? {
		match item_name.as_str().ok_or("Invalid sampler data entry")? {
			"MagFilter" => sampler_create_info.mag_filter = filter_str_to_enum(data.as_str().ok_or("Invalid value in MagFilter")?)?,
			"MinFilter" => sampler_create_info.min_filter = filter_str_to_enum(data.as_str().ok_or("Invalid value in MinFilter")?)?,
			_ => ()
		}
	}

	Ok(Sampler::new(render_pass.device().clone(), sampler_create_info)?)
}

fn prim_topo_str_to_enum(topology_str: &str) -> Result<PrimitiveTopology, Box<dyn std::error::Error>>
{
	match topology_str {
		"PointList" => Ok(PrimitiveTopology::PointList),
		"LineList" => Ok(PrimitiveTopology::LineList),
		"LineStrip" => Ok(PrimitiveTopology::LineStrip),
		"TriangleList" => Ok(PrimitiveTopology::TriangleList),
		"TriangleStrip" => Ok(PrimitiveTopology::TriangleStrip),
		"TriangleFan" => Ok(PrimitiveTopology::TriangleFan),
		"LineListWithAdjacency" => Ok(PrimitiveTopology::LineListWithAdjacency),
		"LineStripWithAdjacency" => Ok(PrimitiveTopology::LineStripWithAdjacency),
		"TriangleListWithAdjacency" => Ok(PrimitiveTopology::TriangleListWithAdjacency),
		"TriangleStripWithAdjacency" => Ok(PrimitiveTopology::TriangleStripWithAdjacency),
		"PatchList" => Ok(PrimitiveTopology::PatchList),
		_ => Err("Invalid primitive topology specified".into())
	}
}

fn filter_str_to_enum(filter_str: &str) -> Result<vulkano::sampler::Filter, Box<dyn std::error::Error>>
{
	match filter_str {
		"Nearest" => Ok(vulkano::sampler::Filter::Nearest),
		"Linear" => Ok(vulkano::sampler::Filter::Linear),
		_ => Err("Invalid sampler filter".into())
	}
}

fn load_spirv(device: Arc<vulkano::device::Device>, filename: &str) 
	-> Result<Arc<vulkano::shader::ShaderModule>, Box<dyn std::error::Error>>
{
	let spv_data = std::fs::read(&filename)?;
	Ok(unsafe { vulkano::shader::ShaderModule::from_bytes(device, &spv_data) }?)
}

/// Load the SPIR-V file, and also automatically determine the given vertex shader's vertex inputs using information from the SPIR-V file.
fn load_spirv_vertex(device: Arc<vulkano::device::Device>, filename: &str) 
	-> Result<(Arc<vulkano::shader::ShaderModule>, VertexInputState), Box<dyn std::error::Error>>
{
	let spv_data = std::fs::read(&filename)?;
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
	match reflect_format {
		ReflectFormat::R32_UINT => Ok(Format::R32_UINT),
		ReflectFormat::R32_SINT => Ok(Format::R32_UINT),
		ReflectFormat::R32_SFLOAT => Ok(Format::R32_SFLOAT),
		ReflectFormat::R32G32_UINT => Ok(Format::R32G32_UINT),
		ReflectFormat::R32G32_SINT => Ok(Format::R32G32_SINT),
		ReflectFormat::R32G32_SFLOAT => Ok(Format::R32G32_SFLOAT),
		ReflectFormat::R32G32B32_UINT => Ok(Format::R32G32B32_UINT),
		ReflectFormat::R32G32B32_SINT => Ok(Format::R32G32B32_SINT),
		ReflectFormat::R32G32B32_SFLOAT => Ok(Format::R32G32B32_SFLOAT),
		ReflectFormat::R32G32B32A32_UINT => Ok(Format::R32G32B32A32_UINT),
		ReflectFormat::R32G32B32A32_SINT => Ok(Format::R32G32B32A32_SINT),
		ReflectFormat::R32G32B32A32_SFLOAT => Ok(Format::R32G32B32A32_SFLOAT),
		_ => Err(UnsupportedVertexInputFormat)
	}
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

