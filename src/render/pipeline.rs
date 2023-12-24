/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::{Format, NumericType};
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
	depth_stencil::{DepthState, DepthStencilState, StencilState},
	input_assembly::{InputAssemblyState, PrimitiveTopology},
	multisample::MultisampleState,
	rasterization::RasterizationState,
	subpass::PipelineRenderingCreateInfo,
	vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState},
	viewport::ViewportState,
	GraphicsPipelineCreateInfo,
};
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::{spirv::ExecutionModel, EntryPoint, ShaderInterfaceEntryType, ShaderModule};

use crate::EngineError;

/// Create a new graphics pipeline using the given parameters.
///
/// NOTE: In the vertex shader, giving vertex inputs a name that ends with '_INSTANCE' will apply
/// `VertexInputRate::Instance` for that input. Otherwise, `VertexInputRate::Vertex` will be used.
pub fn new(
	vk_dev: Arc<Device>,
	primitive_topology: PrimitiveTopology,
	shader_modules: &[Arc<ShaderModule>],
	rasterization_state: RasterizationState,
	pipeline_layout: Arc<PipelineLayout>,
	color_attachments: &[(Format, Option<AttachmentBlend>)],
	depth_attachment: Option<(Format, DepthState)>,
	stencil_attachment: Option<(Format, StencilState)>,
) -> Result<Arc<GraphicsPipeline>, EngineError>
{
	let primitive_restart_enable =
		primitive_topology == PrimitiveTopology::TriangleStrip || primitive_topology == PrimitiveTopology::TriangleFan;
	let input_assembly_state = Some(InputAssemblyState {
		topology: primitive_topology,
		primitive_restart_enable,
		..Default::default()
	});

	let entry_points: Vec<_> = shader_modules
		.iter()
		.map(|sm| {
			sm.entry_point("main")
				.ok_or_else(|| EngineError::from("no 'main' entry point in shader"))
		})
		.collect::<Result<_, _>>()?;

	let vertex_shader = entry_points
		.iter()
		.find(|entry_point| entry_point.info().execution_model == ExecutionModel::Vertex)
		.ok_or_else(|| EngineError::from("pipeline::new: no vertex shader was provided"))?;

	let vertex_input_state = Some(gen_vertex_input_state(&vertex_shader));

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

	// Some notes regarding observed support for depth/stencil formats:
	//
	// - `D16_UNORM`: Supported on all GPUs.
	// - `D16_UNORM_S8_UINT`: Only supported on AMD GPUs.
	// - `X8_D24_UNORM_PACK32`: Only supported on NVIDIA and Intel GPUs.
	// - `D24_UNORM_S8_UINT`: Only supported on NVIDIA and Intel GPUs.
	// - `D32_SFLOAT`: Supported on all GPUs.
	// - `D32_SFLOAT_S8_UINT`: Supported on all GPUs.
	// - `S8_UINT`: Only supported on AMD GPUs. Possibly supported on NVIDIA GPUs.
	//
	// (source: https://vulkan.gpuinfo.org/listoptimaltilingformats.php)
	let (depth_attachment_format, depth_state) = depth_attachment.unzip();
	let (stencil_attachment_format, stencil_state) = stencil_attachment.unzip();

	let depth_stencil_state =
		(depth_attachment_format.is_some() || stencil_attachment_format.is_some()).then(|| DepthStencilState {
			depth: depth_state,
			stencil: stencil_state,
			..Default::default()
		});

	let rendering_info = PipelineRenderingCreateInfo {
		color_attachment_formats,
		depth_attachment_format,
		stencil_attachment_format,
		..Default::default()
	};

	let stages = entry_points
		.into_iter()
		.map(|e| PipelineShaderStageCreateInfo::new(e))
		.collect();

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
		..GraphicsPipelineCreateInfo::layout(pipeline_layout)
	};

	GraphicsPipeline::new(vk_dev.clone(), None, pipeline_info)
		.map_err(|e| EngineError::vulkan_error("failed to create graphics pipeline", e))
}

/// Automatically determine the given vertex shader's vertex inputs using information from the shader module.
fn gen_vertex_input_state(entry_point: &EntryPoint) -> VertexInputState
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

	vertex_input_state
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
