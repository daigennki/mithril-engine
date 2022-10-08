/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::path::Path;
use serde::Deserialize;
use vulkano::descriptor_set::{ PersistentDescriptorSet, WriteDescriptorSet };
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use super::{ Material, DeferMaterialLoading, ColorInput, /*SingleChannelInput*/ };
use crate::GenericEngineError;
use crate::render::{ RenderContext, command_buffer::CommandBuffer };

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
		PBR { base_color: base_color, descriptor_set: None }
	}
}
impl DeferMaterialLoading for PBR
{
	fn update_descriptor_set(&mut self, parent_folder: &Path, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		// TODO: roughness and specular textures
		let base_color_tex = self.base_color.into_texture(parent_folder, render_ctx)?;

		self.descriptor_set = Some(render_ctx.new_descriptor_set(self.pipeline_name(), 2, [
			WriteDescriptorSet::image_view(1, base_color_tex.view())
		])?);

		Ok(())
	}

	fn get_descriptor_set(&self) -> Option<&Arc<PersistentDescriptorSet>>
	{
		self.descriptor_set.as_ref()
	}
}

