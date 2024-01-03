/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::{Format, NumericType};
use vulkano::pipeline::graphics::{
	color_blend::ColorBlendState,
	depth_stencil::DepthStencilState,
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
	topology: PrimitiveTopology,
	shader_modules: &[Arc<ShaderModule>],
	rasterization_state: RasterizationState,
	pipeline_layout: Arc<PipelineLayout>,
	rendering_info: PipelineRenderingCreateInfo,
	color_blend_state: Option<ColorBlendState>,
	depth_stencil_state: Option<DepthStencilState>,
) -> crate::Result<Arc<GraphicsPipeline>>
{
	let input_assembly_state = Some(InputAssemblyState {
		topology,
		primitive_restart_enable: topology == PrimitiveTopology::TriangleStrip || topology == PrimitiveTopology::TriangleFan,
		..Default::default()
	});

	let stages: smallvec::SmallVec<[PipelineShaderStageCreateInfo; 5]> = shader_modules
		.iter()
		.map(|sm| {
			let entry_point = sm.entry_point("main").ok_or("no 'main' entry point in shader")?;
			Ok(PipelineShaderStageCreateInfo::new(entry_point))
		})
		.collect::<Result<_, EngineError>>()?;

	let vertex_input_state = stages
		.iter()
		.find(|stage| stage.entry_point.info().execution_model == ExecutionModel::Vertex)
		.map(|vs_stage| gen_vertex_input_state(&vs_stage.entry_point));

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

	Ok(GraphicsPipeline::new(vk_dev.clone(), None, pipeline_info)?)
}

/// Automatically determine the given vertex shader's vertex inputs using information from the shader module.
fn gen_vertex_input_state(entry_point: &EntryPoint) -> VertexInputState
{
	if entry_point.info().input_interface.elements().len() > 0 {
		log::debug!("Automatically generating VertexInputState:");
	}

	let (bindings, attributes) = entry_point
		.info()
		.input_interface
		.elements()
		.iter()
		.map(|input| {
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
			((binding, binding_desc), (binding, attribute_desc))
		})
		.unzip();

	VertexInputState {
		bindings,
		attributes,
		..Default::default()
	}
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
