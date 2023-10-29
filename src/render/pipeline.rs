/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::error::Error;
use std::sync::Arc;
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
use vulkano::shader::{ShaderInterfaceEntryType, ShaderModule};

use crate::GenericEngineError;

pub fn new(
	vk_dev: Arc<Device>,
	primitive_topology: PrimitiveTopology,
	vs: Arc<ShaderModule>,
	fs_info: Option<(Arc<ShaderModule>, ColorBlendState)>,
	set_layouts: Vec<Arc<DescriptorSetLayout>>,
	push_constant_ranges: Vec<PushConstantRange>,
	rendering_info: PipelineRenderingCreateInfo,
	depth_stencil_state: Option<DepthStencilState>,
) -> Result<Arc<GraphicsPipeline>, GenericEngineError>
{
	let mut stages = Vec::with_capacity(5);

	let input_assembly_state = Some(InputAssemblyState{
		topology: primitive_topology,
		primitive_restart_enable: primitive_topology == PrimitiveTopology::TriangleStrip,
		..Default::default()
	});

	let vertex_input_state = Some(gen_vertex_input_state(vs.clone())?);
	stages.push(get_shader_stage(&vs, "main")?);
	
	// load fragment shader (optional)
	let mut color_blend_state = None;
	if let Some((fs, blend_state)) = fs_info {
		stages.push(get_shader_stage(&fs, "main")?);
		color_blend_state = Some(blend_state);
	}

	let layout_info = PipelineLayoutCreateInfo { set_layouts, push_constant_ranges, ..Default::default() };
	print_pipeline_layout(&layout_info);

	let pipeline_info = GraphicsPipelineCreateInfo {
		stages: stages.into(),
		vertex_input_state,
		input_assembly_state,
		viewport_state: Some(ViewportState::default()),
		rasterization_state: Some(RasterizationState{ cull_mode: CullMode::Back, ..Default::default() }),
		multisample_state: Some(MultisampleState::default()),
		depth_stencil_state,
		color_blend_state,
		dynamic_state: [ DynamicState::Viewport ].into_iter().collect(),
		subpass: Some(rendering_info.into()),
		..GraphicsPipelineCreateInfo::layout(PipelineLayout::new(vk_dev.clone(), layout_info)?)
	};

	Ok(GraphicsPipeline::new(vk_dev.clone(), None, pipeline_info)?)
}

pub fn new_from_config(
	vk_dev: Arc<Device>,
	config: &PipelineConfig,
	set_layouts: Vec<Arc<DescriptorSetLayout>>,
	push_constant_ranges: Vec<PushConstantRange>,
) -> Result<Arc<GraphicsPipeline>, GenericEngineError> 
{
	let rendering_info = PipelineRenderingCreateInfo {
		color_attachment_formats: vec![ Some(Format::R16G16B16A16_SFLOAT), ],
		depth_attachment_format: config.depth_processing.then_some(super::MAIN_DEPTH_FORMAT),
		..Default::default()
	};

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
	let attachment_count = rendering_info.color_attachment_formats.len().try_into().unwrap();
	let color_blend_state = ColorBlendState::with_attachment_states(attachment_count, common_blend_attachment_state);

	let depth_stencil_state = config.depth_processing.then(|| DepthStencilState {
		depth: Some(DepthState {
			write_enable: true,
			compare_op: CompareOp::Less,
		}),
		..Default::default()
	});

	new(
		vk_dev,
		config.primitive_topology,
		config.vertex_shader.clone(),
		Some((config.fragment_shader.clone().unwrap(), color_blend_state)),
		set_layouts,
		push_constant_ranges,
		rendering_info,
		depth_stencil_state,
	)
}

pub fn new_from_config_transparency(
	vk_dev: Arc<Device>,
	config: &PipelineConfig,
	set_layouts: Vec<Arc<DescriptorSetLayout>>,
	push_constant_ranges: Vec<PushConstantRange>,
) -> Result<Arc<GraphicsPipeline>, GenericEngineError>
{
	let rendering_info = PipelineRenderingCreateInfo {
		color_attachment_formats: vec![
			Some(Format::R16G16B16A16_SFLOAT),
			Some(Format::R8_UNORM),
		],
		depth_attachment_format: Some(super::MAIN_DEPTH_FORMAT),
		..Default::default()
	};

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

	let depth_stencil_state = config.depth_processing.then(|| DepthStencilState {
		depth: Some(DepthState {
			write_enable: false,
			compare_op: CompareOp::Less,
		}),
		..Default::default()
	});

	new(
		vk_dev,
		config.primitive_topology,
		config.vertex_shader.clone(),
		Some((config.fragment_shader_transparency.clone().unwrap(), wboit_accum_blend)),
		set_layouts,
		push_constant_ranges,
		rendering_info,
		depth_stencil_state,
	)
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

pub struct PipelineConfig
{
	pub vertex_shader: Arc<ShaderModule>,
	pub fragment_shader: Option<Arc<ShaderModule>>,
	pub fragment_shader_transparency: Option<Arc<ShaderModule>>,
	pub alpha_blending: bool,
	pub primitive_topology: PrimitiveTopology,
	pub depth_processing: bool,
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
fn print_pipeline_layout(layout_info: &PipelineLayoutCreateInfo)
{
	log::debug!("Pipeline layout has:");
	for (i, set) in layout_info.set_layouts.iter().enumerate() {
		let bindings = set.bindings();
		for (binding_i, binding) in bindings {
			log::debug!(
				"set {}, binding {}: {}x {:?} for {:?} shader(s)",
				i,
				binding_i,
				binding.descriptor_count,
				binding.descriptor_type,
				binding.stages,
			);
		}
	}
	for range in layout_info.push_constant_ranges.iter() {
		log::debug!(
			"push constant range at offset {} with size {} for {:?} shader(s)",
			range.offset,
			range.size,
			range.stages
		);
	}
}
