 /* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::path::Path;
use serde::Deserialize;
use vulkano::descriptor_set::{ PersistentDescriptorSet, WriteDescriptorSet };
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use super::{ Material, ColorInput, /*SingleChannelInput*/ };
use crate::GenericEngineError;
use crate::render::{ RenderContext, command_buffer::CommandBuffer };

/// The standard PBR (Physically Based Rendering) material.
#[derive(Deserialize)]
pub struct PBR
{
	base_color: ColorInput,
	//roughness: SingleChannelInput,
	//specular: SingleChannelInput,

	#[serde(skip)]
	descriptor_set: Option<Arc<PersistentDescriptorSet>>,
}
#[typetag::deserialize]
impl Material for PBR
{
	fn pipeline_name(&self) -> &'static str
	{
		"PBR"
	}

	fn update_descriptor_set(&mut self, path_to_this: &Path, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		let tex_path_prefix = path_to_this.parent().map(|p| p.to_path_buf()).unwrap_or_default();

		// TODO: roughness and specular textures
		let base_color_tex = self.base_color.into_texture(&tex_path_prefix, render_ctx)?;

		self.descriptor_set = Some(render_ctx.new_descriptor_set(self.pipeline_name(), 2, [
			WriteDescriptorSet::image_view(1, base_color_tex.view())
		])?);

		Ok(())
	}

	fn bind_descriptor_set(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		cb.bind_descriptor_set(2, self.descriptor_set.as_ref().ok_or("material descriptor set not loaded")?.clone())?;
		Ok(())
	}
}



