 /* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::path::{ PathBuf, Path };
use serde::Deserialize;
use vulkano::format::Format;
use vulkano::image::{ ImageDimensions, MipmapsCount };
use vulkano::descriptor_set::{ PersistentDescriptorSet, WriteDescriptorSet };
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use super::{ Material, /*ColorInput, SingleChannelInput*/ };
use crate::GenericEngineError;
use crate::render::{ RenderContext, command_buffer::CommandBuffer };

/// The standard PBR (Physically Based Rendering) material.
#[derive(Deserialize)]
pub struct PBR
{
	// ColorInput and SingleChannelInput can't be deserialized properly right now for some reason...
	base_color: PathBuf, //ColorInput,
	//roughness: SingleChannelInput,
	//specular: SingleChannelInput,

	#[serde(skip)]
	descriptor_set: Option<Arc<PersistentDescriptorSet>>
}
#[typetag::deserialize]
impl Material for PBR
{
	fn pipeline_name(&self) -> &'static str
	{
		"PBR"
	}

	fn update_descriptor_set(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		// TODO: roughness and specular textures
		
		let base_color_tex = render_ctx.new_texture(&Path::new("models/").join(&self.base_color))?;
		/*let base_color_tex = match &self.base_color {
			ColorInput::Color(color) => {
				// If the input is a single color, make a 1x1 RGBA texture with just the color.
				render_ctx.new_texture_from_iter(
					[color.to_array()], 
					Format::R8G8B8A8_SRGB, 
					ImageDimensions::Dim2d{ width: 1, height: 1, array_layers: 1 },
					MipmapsCount::One
				)?
			},
			ColorInput::Texture(tex_path) => render_ctx.new_texture(&tex_path)?
		};*/

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

