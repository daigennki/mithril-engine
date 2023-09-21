/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use image::{DynamicImage, Rgba};
use rusttype::{point, Font, Scale};
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::format::Format;
use vulkano::image::{ImageDimensions, MipmapsCount};

use super::mesh::Mesh;
use crate::render::RenderContext;
use crate::GenericEngineError;

/// UI component that rasterizes fonts into textures.
#[derive(shipyard::Component)]
#[track(Insertion)]
pub struct Text
{
	font: Font<'static>,
	size: f32,
	quad: Option<Mesh>,
	text_str: String,
}
impl Text
{
	pub fn new(render_ctx: &mut RenderContext, text_str: String, size: f32) -> Result<Self, GenericEngineError>
	{
		// TODO: preload fonts
		let font_data = include_bytes!("../../../resource/mplus-1m-medium.ttf");
		let font = Font::try_from_bytes(font_data as &[u8]).ok_or("Error constructing font")?;

		let mut new_text = Text {
			font,
			size,
			quad: None,
			text_str: text_str.clone(),
		};

		if text_str.is_empty() {
			return Ok(new_text);
		}

		new_text.set_text(text_str, render_ctx)?;
		Ok(new_text)
	}

	pub fn set_text(&mut self, text: String, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		self.text_str = text;

		let scale_uniform = Scale::uniform(self.size);
		let color = (255, 255, 0);
		let v_metrics = self.font.v_metrics(scale_uniform);

		// lay out the glyphs in a line with 1 pixel padding
		let glyphs: Vec<_> = self
			.font
			.layout(&self.text_str, scale_uniform, point(1.0, 1.0 + v_metrics.ascent))
			.collect();

		// work out the layout size
		let glyphs_height = (v_metrics.ascent - v_metrics.descent).ceil() as u32;
		let min_x = glyphs
			.first()
			.ok_or("there were no glyphs for the string!")?
			.pixel_bounding_box()
			.ok_or("pixel_bounding_box was `None`!")?
			.min
			.x;
		let max_x = glyphs
			.last()
			.ok_or("there were no glyphs for the string!")?
			.pixel_bounding_box()
			.ok_or("pixel_bounding_box was `None`!")?
			.max
			.x;
		let glyphs_width = (max_x - min_x) as u32;

		// Create a new rgba image
		let mut image = DynamicImage::new_rgba8(glyphs_width + 2, glyphs_height + 2).into_rgba8();

		// Loop through the glyphs in the text, positing each one on a line
		for glyph in glyphs {
			if let Some(bounding_box) = glyph.pixel_bounding_box() {
				// Draw the glyph into the image per-pixel by using the draw closure
				glyph.draw(|x, y, v| {
					// Offset the position by the glyph bounding box
					let x_offset = x + bounding_box.min.x as u32;
					let y_offset = y + bounding_box.min.y as u32;
					// Make sure the pixel isn't out of bounds. If it is OoB, then don't draw it.
					if x_offset >= image.width() || y_offset >= image.height() {
						log::warn!(
							"Text pixel at ({},{}) is out of bounds of ({},{})",
							x_offset,
							y_offset,
							image.width(),
							image.height()
						);
					} else {
						// Turn the coverage into an alpha value
						image.put_pixel(x_offset, y_offset, Rgba([color.0, color.1, color.2, (v * 255.0) as u8]))
					}
				});
			}
		}

		let img_dim = ImageDimensions::Dim2d {
			width: image.width(),
			height: image.height(),
			array_layers: 1,
		};
		let tex = render_ctx.new_texture_from_iter(image.into_raw(), Format::R8G8B8A8_SRGB, img_dim, MipmapsCount::One)?;

		let mesh_top_left = Vec2::new(img_dim.width() as f32 / -2.0, -v_metrics.ascent - 1.0);
		let mesh_bottom_right = Vec2::new(img_dim.width() as f32 / 2.0, -v_metrics.descent + 1.0);

		self.quad = Some(Mesh::new_from_corners(
			render_ctx,
			mesh_top_left,
			mesh_bottom_right,
			Arc::new(tex),
		)?);

		Ok(())
	}

	/// Obtain the currently displayed string.
	pub fn cur_str(&self) -> String
	{
		self.text_str.clone()
	}

	pub fn draw(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		match self.quad.as_ref() {
			Some(q) => q.draw(cb),
			None => Ok(()),
		}
	}
}
