/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::device::Device;
use vulkano::format::{Format, NumericType};
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState},
	depth_stencil::{CompareOp, DepthState, DepthStencilState},
	input_assembly::{InputAssemblyState, PrimitiveTopology},
	multisample::MultisampleState,
	rasterization::{CullMode, RasterizationState},
	subpass::PipelineRenderingCreateInfo,
	vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState},
	viewport::ViewportState,
	GraphicsPipelineCreateInfo,
};
use vulkano::pipeline::{
	layout::{PipelineLayoutCreateInfo, PushConstantRange},
	DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::{spirv::ExecutionModel, EntryPoint, ShaderInterfaceEntryType, ShaderModule};

use crate::GenericEngineError;

/// Create a new graphics pipeline using the given parameters.
///
/// NOTE: In the vertex shader, giving vertex inputs a name that ends with '_INSTANCE' will apply
/// `VertexInputRate::Instance` for that input. Otherwise, `VertexInputRate::Vertex` will be used.
pub fn new(
	vk_dev: Arc<Device>,
	primitive_topology: PrimitiveTopology,
	shader_modules: &[Arc<ShaderModule>],
	rasterization_state: RasterizationState,
	set_layouts: Vec<Arc<DescriptorSetLayout>>,
	color_attachments: &[(Format, Option<AttachmentBlend>)],
	depth_attachment: Option<(Format, DepthStencilState)>,
) -> Result<Arc<GraphicsPipeline>, GenericEngineError>
{
	let primitive_restart_enable =
		primitive_topology == PrimitiveTopology::TriangleStrip || primitive_topology == PrimitiveTopology::TriangleFan;
	let input_assembly_state = Some(InputAssemblyState {
		topology: primitive_topology,
		primitive_restart_enable,
		..Default::default()
	});

	let mut stages = smallvec::SmallVec::<[PipelineShaderStageCreateInfo; 5]>::new();
	for sm in shader_modules {
		stages.push(get_shader_stage(&sm, "main")?);
	}

	let vertex_shader = stages
		.iter()
		.find(|stage| stage.entry_point.info().execution_model == ExecutionModel::Vertex)
		.ok_or("pipeline::new: no vertex shader was provided")?;

	let vertex_input_state = Some(gen_vertex_input_state(&vertex_shader.entry_point)?);

	// Go through all the stages that have a push constant, combining the shader stage flags
	// and getting the smallest range size to create a single `PushConstantRange`.
	let push_constant_range = stages
		.iter()
		.filter_map(|stage| stage.entry_point.info().push_constant_requirements)
		.reduce(|acc, stage_range| PushConstantRange {
			stages: acc.stages.union(stage_range.stages),
			offset: 0,
			size: acc.size.min(stage_range.size),
		});

	let layout_info = PipelineLayoutCreateInfo {
		set_layouts,
		push_constant_ranges: push_constant_range.into_iter().collect(),
		..Default::default()
	};
	print_pipeline_layout(&layout_info);

	let (color_attachment_formats, color_blend_attachment_states) = color_attachments
		.iter()
		.map(|(format, blend)| {
			let blend_attachment_state = ColorBlendAttachmentState {
				blend: *blend,
				..Default::default()
			};
			(Some(*format), blend_attachment_state)
		})
		.unzip();

	let color_blend_state = (!color_attachments.is_empty()).then(|| ColorBlendState {
		attachments: color_blend_attachment_states,
		..Default::default()
	});

	let (depth_attachment_format, depth_stencil_state) = depth_attachment.unzip();

	let rendering_info = PipelineRenderingCreateInfo {
		color_attachment_formats,
		depth_attachment_format,
		..Default::default()
	};

	let pipeline_info = GraphicsPipelineCreateInfo {
		stages,
		vertex_input_state,
		input_assembly_state,
		viewport_state: Some(ViewportState::default()),
		rasterization_state: Some(rasterization_state),
		multisample_state: Some(MultisampleState::default()),
		depth_stencil_state,
		color_blend_state,
		dynamic_state: [DynamicState::Viewport].into_iter().collect(),
		subpass: Some(rendering_info.into()),
		..GraphicsPipelineCreateInfo::layout(PipelineLayout::new(vk_dev.clone(), layout_info)?)
	};

	Ok(GraphicsPipeline::new(vk_dev.clone(), None, pipeline_info)?)
}

/// Create a pipeline for a material.
pub fn new_for_material(
	vk_dev: Arc<Device>,
	vertex_shader: Arc<ShaderModule>,
	fragment_shader: Arc<ShaderModule>,
	attachment_blend: Option<AttachmentBlend>,
	primitive_topology: PrimitiveTopology,
	set_layouts: Vec<Arc<DescriptorSetLayout>>,
) -> Result<Arc<GraphicsPipeline>, GenericEngineError>
{
	let rasterization_state = RasterizationState {
		cull_mode: CullMode::Back,
		..Default::default()
	};
	let depth_stencil_state = DepthStencilState {
		depth: Some(DepthState::simple()),
		..Default::default()
	};

	new(
		vk_dev,
		primitive_topology,
		&[vertex_shader, fragment_shader],
		rasterization_state,
		set_layouts,
		&[(Format::R16G16B16A16_SFLOAT, attachment_blend)],
		Some((super::MAIN_DEPTH_FORMAT, depth_stencil_state)),
	)
}

/// Create a pipeline for a material, assuming that the fragment shader is specfically for OIT.
pub fn new_for_material_transparency(
	vk_dev: Arc<Device>,
	vertex_shader: Arc<ShaderModule>,
	fragment_shader: Arc<ShaderModule>,
	primitive_topology: PrimitiveTopology,
	set_layouts: Vec<Arc<DescriptorSetLayout>>,
) -> Result<Arc<GraphicsPipeline>, GenericEngineError>
{
	let rasterization_state = RasterizationState {
		cull_mode: CullMode::Back,
		..Default::default()
	};
	let depth_stencil_state = DepthStencilState {
		depth: Some(DepthState {
			write_enable: false,
			compare_op: CompareOp::Less,
		}),
		..Default::default()
	};

	let accum_blend = AttachmentBlend {
		alpha_blend_op: BlendOp::Add,
		..AttachmentBlend::additive()
	};
	let revealage_blend = AttachmentBlend {
		color_blend_op: BlendOp::Add,
		src_color_blend_factor: BlendFactor::Zero,
		dst_color_blend_factor: BlendFactor::OneMinusSrcColor,
		..Default::default()
	};
	let color_attachments = [
		(Format::R16G16B16A16_SFLOAT, Some(accum_blend)),
		(Format::R8_UNORM, Some(revealage_blend)),
	];

	new(
		vk_dev,
		primitive_topology,
		&[vertex_shader, fragment_shader],
		rasterization_state,
		set_layouts,
		&color_attachments,
		Some((super::MAIN_DEPTH_FORMAT, depth_stencil_state)),
	)
}

fn get_shader_stage(
	shader_module: &Arc<ShaderModule>,
	entry_point_name: &str,
) -> Result<PipelineShaderStageCreateInfo, GenericEngineError>
{
	let entry_point = shader_module.entry_point(entry_point_name).ok_or(format!(
		"No entry point called '{}' found in SPIR-V module!",
		entry_point_name
	))?;

	Ok(PipelineShaderStageCreateInfo::new(entry_point))
}

/// Automatically determine the given vertex shader's vertex inputs using information from the shader module.
fn gen_vertex_input_state(entry_point: &EntryPoint) -> Result<VertexInputState, GenericEngineError>
{
	log::debug!("Automatically generating VertexInputState:");
	let vertex_input_state =
		entry_point
			.info()
			.input_interface
			.elements()
			.iter()
			.fold(VertexInputState::new(), |accum, input| {
				let binding = input.location;
				let format = format_from_interface_type(&input.ty);
				let stride = format.components().iter().fold(0, |acc, c| acc + (*c as u32)) / 8;
				let name = input.name.as_ref().map(|n| n.as_ref()).unwrap_or("[unknown]");

				log::debug!("- binding + attribute {binding} '{name}': {format:?} (stride {stride})");

				// If the input name ends with "_INSTANCE", use `VertexInputRate::Instance` for that input.
				let input_rate = name
					.ends_with("_INSTANCE")
					.then_some(VertexInputRate::Instance { divisor: 1 })
					.unwrap_or(VertexInputRate::Vertex);

				let binding_desc = VertexInputBindingDescription { stride, input_rate };
				let attribute_desc = VertexInputAttributeDescription {
					binding,
					format,
					offset: 0,
				};
				accum.binding(binding, binding_desc).attribute(binding, attribute_desc)
			});

	if vertex_input_state.attributes.is_empty() {
		log::debug!("- (empty)");
	}

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
				"- set {}, binding {}: {}x {:?} for {:?} shader(s)",
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
			"- push constant range at offset {} with size {} for {:?} shader(s)",
			range.offset,
			range.size,
			range.stages
		);
	}
}
