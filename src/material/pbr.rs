/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use std::path::Path;
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	WriteDescriptorSet,
};
use vulkano::pipeline::graphics::input_assembly::PrimitiveTopology;
use vulkano::shader::ShaderStages;

use super::{ColorInput, /*SingleChannelInput,*/ Material};
use crate::render::{RenderContext, pipeline::StaticPipelineConfig};
use crate::GenericEngineError;

pub static PIPELINE_CONFIG: StaticPipelineConfig = StaticPipelineConfig {
	vertex_shader: include_bytes!("../../shaders/basic_3d.vert.spv"),
	fragment_shader: Some(include_bytes!("../../shaders/pbr.frag.spv")),
	fragment_shader_transparency: Some(include_bytes!("../../shaders/pbr_mboit_weights.frag.spv")),
	always_pass_depth_test: false,
	alpha_blending: false,
	primitive_topology: PrimitiveTopology::TriangleList,
};

/// The standard PBR (Physically Based Rendering) material.
#[derive(Deserialize)]
pub struct PBR
{
	pub base_color: ColorInput,

	// TODO: roughness and specular textures
	//pub roughness: SingleChannelInput,

	//pub specular: SingleChannelInput,

	#[serde(default)]
	pub transparent: bool,
}
impl PBR
{
	pub fn set_layout_info_pbr(render_ctx: &RenderContext) -> DescriptorSetLayoutCreateInfo
	{
		let bindings = [
			(0, DescriptorSetLayoutBinding { // binding 0: sampler0
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![ render_ctx.get_default_sampler().clone() ],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
			}),
			(1, DescriptorSetLayoutBinding { // binding 1: base_color
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
			}),
		];
		DescriptorSetLayoutCreateInfo {
			bindings: bindings.into(),
			..Default::default()
		}
	}
}

#[typetag::deserialize]
impl Material for PBR
{
	fn material_name(&self) -> &'static str
	{
		"PBR"
	}

	// NOTE: the descriptor set is expected to be bound to set 1
	fn gen_descriptor_set_writes(
		&self,
		parent_folder: &Path,
		render_ctx: &mut RenderContext
	) -> Result<Vec<WriteDescriptorSet>, GenericEngineError>
	{
		let base_color = self.base_color.into_texture(parent_folder, render_ctx)?;

		let writes = vec![ WriteDescriptorSet::image_view(1, base_color.view()) ];

		Ok(writes)
	}

	fn set_layout_info(&self, render_ctx: &RenderContext) -> DescriptorSetLayoutCreateInfo
	{
		Self::set_layout_info_pbr(render_ctx)
	}

	fn has_transparency(&self) -> bool
	{
		self.transparent
	}
}
/*impl DeferMaterialLoading for PBR
{
	fn get_base_color(&self) -> Option<Vec4>
	{
		match &self.base_color {
			ColorInput::Color(c) => Some(*c),
			_ => None,
		}
	}
	fn set_base_color(&mut self, color: Vec4, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		self.base_color = ColorInput::Color(color);
		self.update_descriptor_set(Path::new("./"), render_ctx)
	}
}*/

