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
				// We convert it to sRGB first though, since there's no f32 format for sRGB.
				let linear_color = [
					srgb_to_linear(color.x),
					srgb_to_linear(color.y),
					srgb_to_linear(color.z),
					color.w
				];
				render_ctx.new_texture_from_iter(
					linear_color, 
					Format::R32G32B32A32_SFLOAT, 
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

/// Convert non-linear sRGB color into linear sRGB.
fn srgb_to_linear(c: f32) -> f32
{
	if c <= 0.04045 {
		c / 12.92
	} else {
		((c + 0.055) / 1.055).powf(2.4)
	}
}

