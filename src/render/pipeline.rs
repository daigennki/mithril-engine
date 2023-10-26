/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::error::Error;
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::device::Device;
use vulkano::format::{Format, NumericType};
use vulkano::pipeline::{
	layout::{PipelineLayoutCreateInfo, PushConstantRange}, DynamicState, GraphicsPipeline, PipelineLayout,
	PipelineShaderStageCreateInfo
};
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState, ColorBlendAttachmentState},
	depth_stencil::{CompareOp, DepthState, DepthStencilState},
	input_assembly::{InputAssemblyState, PrimitiveTopology},
	multisample::MultisampleState,
	rasterization::{CullMode, RasterizationState},
	vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState},
	viewport::ViewportState,
	subpass::PipelineRenderingCreateInfo,
	GraphicsPipelineCreateInfo,
};
use vulkano::shader::{ShaderInterfaceEntryType, ShaderModule, ShaderModuleCreateInfo};

use crate::GenericEngineError;

pub struct Pipeline
{
	pipeline: Arc<GraphicsPipeline>,
}
impl Pipeline
{
	pub fn new_from_binary(
		vk_dev: Arc<Device>,
		primitive_topology: PrimitiveTopology,
		vs_bin: &[u8],
		fs_info: Option<(&[u8], ColorBlendState)>,
		set_layouts: Vec<Arc<DescriptorSetLayout>>,
		push_constant_ranges: Vec<PushConstantRange>,
		rendering_info: PipelineRenderingCreateInfo,
		depth_op: CompareOp, 
		depth_write: bool,
	) -> Result<Self, GenericEngineError>
	{
		let vs = load_spirv_bytes(vk_dev.clone(), vs_bin)?;
		let vertex_input_state = Some(gen_vertex_input_state(vs.clone())?);

		let input_assembly_state = Some(InputAssemblyState{
			topology: primitive_topology,
			primitive_restart_enable: primitive_topology == PrimitiveTopology::TriangleStrip,
			..Default::default()
		});
		
		let depth_stencil_state = DepthStencilState {
			depth: Some(DepthState {
				write_enable: depth_write,
				compare_op: depth_op,
			}),
			..Default::default()
		};

		let rasterization_state = Some(RasterizationState{ cull_mode: CullMode::Back, ..Default::default() });

		let mut stages = Vec::with_capacity(5);
		stages.push(get_shader_stage(&vs, "main")?);
		
		// load fragment shader (optional)
		let mut color_blend_state = None;
		if let Some((fs_bin, blend_state)) = fs_info {
			// load the fragment shader for the opaque pass
			let fs = load_spirv_bytes(vk_dev.clone(), fs_bin)?;
			stages.push(get_shader_stage(&fs, "main")?);
			color_blend_state = Some(blend_state);
		}

		// create the pipeline layout
		let layout_info = PipelineLayoutCreateInfo {
			set_layouts: set_layouts.clone(),
			push_constant_ranges: push_constant_ranges.clone(),
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(vk_dev.clone(), layout_info)?;
		let pipeline_info = GraphicsPipelineCreateInfo {
			stages: stages.into(),
			vertex_input_state: vertex_input_state.clone(),
			input_assembly_state: input_assembly_state.clone(),
			viewport_state: Some(ViewportState::default()),
			rasterization_state: rasterization_state.clone(),
			multisample_state: Some(MultisampleState::default()),
			depth_stencil_state: Some(depth_stencil_state.clone()),
			color_blend_state,
			dynamic_state: [ DynamicState::Viewport ].into_iter().collect(),
			subpass: Some(rendering_info.into()),
			..GraphicsPipelineCreateInfo::layout(pipeline_layout)
		};
	
		// create the pipeline
		let pipeline = GraphicsPipeline::new(vk_dev.clone(), None, pipeline_info)?;
		print_pipeline_descriptors_info(pipeline.as_ref());

		Ok(Pipeline {
			pipeline,
		})
	}

	pub fn new_from_config(
		vk_dev: Arc<Device>,
		config: &StaticPipelineConfig,
		rendering_info: PipelineRenderingCreateInfo,
		set_layouts: Vec<Arc<DescriptorSetLayout>>,
		push_constant_ranges: Vec<PushConstantRange>,
		depth_write: bool,
		is_for_transparency: bool,
	) -> Result<Self, GenericEngineError> 
	{
		let attachment_count = rendering_info.color_attachment_formats.len().try_into().unwrap();

		let fs_info = if is_for_transparency {
			let mut wboit_accum_blend = ColorBlendState::with_attachment_states(2, ColorBlendAttachmentState {
				blend: Some(AttachmentBlend {
					alpha_blend_op: BlendOp::Add,
					..AttachmentBlend::additive()
				}),
				..Default::default()
			});
			wboit_accum_blend.attachments[1].blend = Some(AttachmentBlend {
				color_blend_op: BlendOp::Add,
				src_color_blend_factor: BlendFactor::Zero,
				dst_color_blend_factor: BlendFactor::OneMinusSrcColor,
				..AttachmentBlend::ignore_source()
			});

			(config.fragment_shader_transparency.unwrap(), wboit_accum_blend)
		} else {
			let common_blend_attachment_state = if config.fragment_shader_transparency.is_some() {
				// Disable blending if this is an opaque rendering pipeline, 
				// and transparency will be handled in a separate pass.
				ColorBlendAttachmentState::default()
			} else {
				ColorBlendAttachmentState {
					blend: config.alpha_blending.then_some(AttachmentBlend::alpha()),
					..Default::default()
				}
			};
			let color_blend_state = ColorBlendState::with_attachment_states(attachment_count, common_blend_attachment_state);

			(config.fragment_shader.unwrap(), color_blend_state)
		};

		let depth_op = config
			.always_pass_depth_test
			.then_some(CompareOp::Always)
			.unwrap_or(CompareOp::Less);

		Pipeline::new_from_binary(
			vk_dev,
			config.primitive_topology,
			config.vertex_shader,
			Some(fs_info),
			set_layouts,
			push_constant_ranges,
			rendering_info,
			depth_op,
			depth_write,
		)
	}

	pub fn bind<L>(&self, command_buffer: &mut AutoCommandBufferBuilder<L>) -> Result<(), GenericEngineError>
	{
		command_buffer.bind_pipeline_graphics(self.pipeline.clone())?;
		Ok(())
	}

	pub fn layout(&self) -> Arc<PipelineLayout>
	{
		let pipeline_ref: &dyn vulkano::pipeline::Pipeline = self.pipeline.as_ref();
		pipeline_ref.layout().clone()
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

#[derive(Default)]
pub struct StaticPipelineConfig
{
	pub vertex_shader: &'static [u8],
	pub fragment_shader: Option<&'static [u8]>,
	pub fragment_shader_transparency: Option<&'static [u8]>,
	pub always_pass_depth_test: bool,
	pub alpha_blending: bool,
	pub primitive_topology: PrimitiveTopology,
}

fn get_shader_stage(
	shader_module: &Arc<ShaderModule>,
	entry_point_name: &str,
) -> Result<PipelineShaderStageCreateInfo, GenericEngineError>
{
	let entry_point = shader_module
		.entry_point(entry_point_name)
		.ok_or(format!("No entry point called '{}' found in SPIR-V module!", entry_point_name))?;

	Ok(PipelineShaderStageCreateInfo::new(entry_point))
}

fn load_spirv_bytes(device: Arc<vulkano::device::Device>, bytes: &[u8]) -> Result<Arc<ShaderModule>, GenericEngineError>
{
	let spv_words = vulkano::shader::spirv::bytes_to_words(&bytes)?;
	Ok(unsafe { ShaderModule::new(device, ShaderModuleCreateInfo::new(&spv_words)) }?)
}

/// Automatically determine the given vertex shader's vertex inputs using information from the shader module.
fn gen_vertex_input_state(shader_module: Arc<ShaderModule>) -> Result<VertexInputState, GenericEngineError>
{
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

	log::debug!("Automatically generated VertexInputState: {:#?}", &vertex_input_state);

	Ok(vertex_input_state)
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
