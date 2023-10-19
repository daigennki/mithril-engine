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
use vulkano::shader::ShaderStages;

use super::{ColorInput, /*SingleChannelInput,*/ Material};
use crate::render::RenderContext;
use crate::GenericEngineError;

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

#[typetag::deserialize]
impl Material for PBR
{
	fn pipeline_name(&self) -> &'static str
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

