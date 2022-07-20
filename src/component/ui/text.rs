/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use vulkano::command_buffer::DrawError;
use vulkano::format::Format;
use vulkano::image::{ ImageDimensions, MipmapsCount };
use image::{ DynamicImage, Rgba };
use rusttype::{ point, Font, Scale };
use glam::*;
use super::mesh::Mesh;
use crate::render::RenderContext;

/// UI component that rasterizes fonts into textures.
pub struct Text
{
	quad: Option<Mesh>,
	cur_str: String
}
impl Text
{
	pub fn new(render_ctx: &mut RenderContext, text_str: &str, size: f32) -> Result<Text, Box<dyn std::error::Error>>
	{
		if text_str.is_empty() {
			return Ok(Text{ quad: None, cur_str: text_str.to_string() })
		}

		// TODO: preload fonts
		let font_data = include_bytes!("../../../examples/ui_menu/resource/mplus-1m-medium.ttf");
		let font = Font::try_from_bytes(font_data as &[u8]).ok_or("Error constructing font")?;

		let scale_uniform = Scale::uniform(size);
		let color = (255, 255, 255);
		let v_metrics = font.v_metrics(scale_uniform);

		// lay out the glyphs in a line with 1 pixel padding
		let glyphs: Vec<_> = font.layout(text_str, scale_uniform, point(1.0, 1.0 + v_metrics.ascent)).collect();

		// work out the layout size
		let glyphs_height = (v_metrics.ascent - v_metrics.descent).ceil() as u32;
		let min_x = glyphs.first().ok_or("there were no glyphs for the string!")?
			.pixel_bounding_box().ok_or("pixel_bounding_box was `None`!")?.min.x;
		let max_x = glyphs.last().ok_or("there were no glyphs for the string!")?
			.pixel_bounding_box().ok_or("pixel_bounding_box was `None`!")?.max.x;
		let glyphs_width = (max_x - min_x) as u32;
		
		// Create a new rgba image
		let mut image = DynamicImage::new_rgba8(glyphs_width + 2, glyphs_height + 2).into_rgba8();

		// Loop through the glyphs in the text, positing each one on a line
		for glyph in glyphs {
			match glyph.pixel_bounding_box() {
				// Draw the glyph into the image per-pixel by using the draw closure
				Some(bounding_box) => glyph.draw(|x, y, v| {
					// Offset the position by the glyph bounding box
					let x_offset = x + bounding_box.min.x as u32;
					let y_offset = y + bounding_box.min.y as u32;
					// Make sure the pixel isn't out of bounds. If it is OoB, then don't draw it.
					if x_offset >= image.width() || y_offset >= image.height() {
						log::warn!(
							"Text pixel at ({},{}) is out of bounds of ({},{})", 
							x_offset, y_offset, image.width(), image.height()
						);
					} else {
						// Turn the coverage into an alpha value
						image.put_pixel(x_offset, y_offset, Rgba([color.0, color.1, color.2, (v * 255.0) as u8]))
					}
					
				}),
				None => ()
			}
		}

		let img_dim = ImageDimensions::Dim2d{ width: image.width(), height: image.height(), array_layers: 1 };
		let tex = render_ctx.new_texture_from_iter(image.into_raw(), Format::R8G8B8A8_SRGB, img_dim, MipmapsCount::One)?;

		let mesh_top_left = Vec2::new(img_dim.width() as f32 / -2.0, -v_metrics.ascent - 1.0);
		let mesh_bottom_right = Vec2::new(img_dim.width() as f32 / 2.0, -v_metrics.descent + 1.0);

		Ok(Text{
			quad: Some(Mesh::new_from_corners(render_ctx, mesh_top_left, mesh_bottom_right, tex)?),
			cur_str: text_str.to_string()
		})
	}

	/// Obtain the currently displayed string.
	pub fn cur_str(&self) -> String
	{
		self.cur_str.clone()
	}

	pub fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		match self.quad.as_ref() {
			Some(q) => q.draw(render_ctx),
			None => Ok(())
		}
	}
}
