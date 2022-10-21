/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use glam::*;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};

use super::{ColorInput /*SingleChannelInput*/, DeferMaterialLoading, Material};
use crate::render::RenderContext;
use crate::GenericEngineError;

/// The standard PBR (Physically Based Rendering) material.
#[derive(Deserialize, Material)]
pub struct PBR
{
	base_color: ColorInput,
	//roughness: SingleChannelInput,
	//specular: SingleChannelInput,
	#[serde(skip)]
	descriptor_set: Option<Arc<PersistentDescriptorSet>>,
}
impl PBR
{
	pub fn new(base_color: ColorInput) -> Self
	{
		PBR { base_color, descriptor_set: None }
	}
}
impl DeferMaterialLoading for PBR
{
	fn update_descriptor_set(&mut self, parent_folder: &Path, render_ctx: &mut RenderContext)
		-> Result<(), GenericEngineError>
	{
		// TODO: roughness and specular textures
		let base_color_tex = self.base_color.into_texture(parent_folder, render_ctx)?;

		self.descriptor_set = Some(
			render_ctx
				.new_descriptor_set(self.pipeline_name(), 2, [WriteDescriptorSet::image_view(1, base_color_tex.view())])?,
		);

		Ok(())
	}

	fn get_descriptor_set(&self) -> Option<&Arc<PersistentDescriptorSet>>
	{
		self.descriptor_set.as_ref()
	}

	fn get_base_color(&self) -> Vec4
	{
		match &self.base_color {
			ColorInput::Color(c) => *c,
			ColorInput::Texture(t) => Vec4::new(1.0, 1.0, 1.0, 1.0),
		}
	}
	fn set_base_color(&mut self, color: Vec4, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		self.base_color = ColorInput::Color(color);
		self.update_descriptor_set(Path::new("./"), render_ctx)
	}
}
