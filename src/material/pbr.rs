/* -----------------------------------------------------------------------------
 	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};

use super::{ColorInput /*SingleChannelInput*/, DeferMaterialLoading, Material};
use crate::render::RenderContext;
use crate::GenericEngineError;

/// The standard PBR (Physically Based Rendering) material.
#[derive(Deserialize)]
pub struct PBR
{
	base_color: ColorInput,
	//roughness: SingleChannelInput,
	//specular: SingleChannelInput,
	#[serde(default)]
	transparent: bool,

	#[serde(skip)]
	descriptor_set: Option<Arc<PersistentDescriptorSet>>,
}
impl PBR
{
	pub fn new(base_color: ColorInput, transparent: bool) -> Self
	{
		PBR {
			base_color,
			transparent,
			descriptor_set: None,
		}
	}
}
#[typetag::deserialize]
impl Material for PBR
{
	fn pipeline_name(&self) -> &'static str
	{
		"PBR"
	}

	fn bind_descriptor_set(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
	) -> Result<(), GenericEngineError>
	{
		crate::render::bind_descriptor_set(
			cb,
			2,
			self.get_descriptor_set().ok_or("material descriptor set not loaded")?.clone(),
		)?;
		Ok(())
	}

	fn has_transparency(&self) -> bool
	{
		self.transparent
	}
}
impl DeferMaterialLoading for PBR
{
	fn update_descriptor_set(&mut self, parent_folder: &Path, render_ctx: &mut RenderContext)
		-> Result<(), GenericEngineError>
	{
		// TODO: roughness and specular textures
		let base_color_tex = self.base_color.into_texture(parent_folder, render_ctx)?;

		self.descriptor_set = Some(render_ctx.new_descriptor_set(
			self.pipeline_name(),
			2,
			[WriteDescriptorSet::image_view(1, base_color_tex.view())],
		)?);

		Ok(())
	}

	fn get_descriptor_set(&self) -> Option<&Arc<PersistentDescriptorSet>>
	{
		self.descriptor_set.as_ref()
	}

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
}
