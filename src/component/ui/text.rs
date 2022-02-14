/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use vulkano::command_buffer::DrawError;
use vulkano::format::Format;
use vulkano::image::{ ImageDimensions, MipmapsCount };
use image::{ DynamicImage, Rgba };
use rusttype::{ point, Font, Scale };
use super::mesh::Mesh;
use crate::render::RenderContext;

/// UI component that rasterizes fonts into textures.
pub struct Text
{
	quad: Mesh
}
impl Text
{
	pub fn new(render_ctx: &mut RenderContext, text_str: &str) -> Result<Text, Box<dyn std::error::Error>>
	{
		let font_data = include_bytes!("../../../examples/ui_menu/resource/mplus-1m-medium.ttf");
		let font = Font::try_from_bytes(font_data as &[u8]).ok_or("Error constructing font")?;

		let scale = Scale::uniform(32.0);
		let color = (255, 255, 255);
		let v_metrics = font.v_metrics(scale);

		// lay out the glyphs in a line with 20 pixels padding
		let glyphs: Vec<_> = font.layout(text_str, scale, point(20.0, 20.0 + v_metrics.ascent)).collect();

		// work out the layout size
		let glyphs_height = (v_metrics.ascent - v_metrics.descent).ceil() as u32;
		let min_x = glyphs.first().ok_or("there were no glyphs for the string!")?
			.pixel_bounding_box().ok_or("pixel_bounding_box was none!")?.min.x;
		let max_x = glyphs.last().ok_or("there were no glyphs for the string!")?
			.pixel_bounding_box().ok_or("pixel_bounding_box was none!")?.max.x;
		let glyphs_width = (max_x - min_x) as u32;
		
		// Create a new rgba image with some padding
		let mut image = DynamicImage::new_rgba8(glyphs_width + 40, glyphs_height + 40).into_rgba8();

		// Loop through the glyphs in the text, positing each one on a line
		for glyph in glyphs {
			if let Some(bounding_box) = glyph.pixel_bounding_box() {
				// Draw the glyph into the image per-pixel by using the draw closure
				glyph.draw(|x, y, v| {
					image.put_pixel(
						// Offset the position by the glyph bounding box
						x + bounding_box.min.x as u32,
						y + bounding_box.min.y as u32,
						// Turn the coverage into an alpha value
						Rgba([color.0, color.1, color.2, (v * 255.0) as u8])
					)
				})
			}
		}

		let img_dim = ImageDimensions::Dim2d{ width: image.width(), height: image.height(), array_layers: 1 };
		let tex = render_ctx.new_texture_from_iter(image.into_raw(), Format::R8G8B8A8_SRGB, img_dim, MipmapsCount::One)?;

		Ok(Text{
			quad: Mesh::new(render_ctx, tex)?
		})
	}

	pub fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), DrawError>
	{
		self.quad.draw(render_ctx)
	}
}