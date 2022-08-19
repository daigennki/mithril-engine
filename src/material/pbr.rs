 /* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::path::Path;
use glam::*;
use serde::Deserialize;
use vulkano::format::Format;
use vulkano::image::{ ImageDimensions, MipmapsCount };
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
		// TODO: roughness and specular textures
		let base_color_tex = match &self.base_color {
			ColorInput::Color(color) => {
				// If the input is a single color, make a 1x1 RGBA texture with just the color.
				// The input is in f32 format though, so we need to convert it to u8 first.
				render_ctx.new_texture_from_iter(
					(*color * u8::MAX as f32).round().to_array().iter().map(|f| *f as u8), 
					Format::R8G8B8A8_SRGB, 
					ImageDimensions::Dim2d{ width: 1, height: 1, array_layers: 1 },
					MipmapsCount::One
				)?
			},
			ColorInput::Texture(tex_path) => render_ctx.new_texture(
				&path_to_this.parent().unwrap_or_else(|| Path::new("./models/")).join(tex_path)
			)?
		};

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

