/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */

use vulkano::command_buffer::DrawError;
use crate::rendercontext::RenderContext;
use super::quad::Quad;

pub struct Img
{
	quad: Quad
}
impl Img
{
	pub fn new(render_ctx: &mut RenderContext, path: &std::path::Path) 
		-> Result<Img, Box<dyn std::error::Error>>
	{
		let tex = render_ctx.new_texture(path)?;
		let quad = Quad::new(render_ctx, tex)?;

		Ok(Img{
			quad: quad
		})
	}

	pub fn tex_dimensions(&self) -> [u32; 2]
	{
		self.quad.tex_dimensions()
	}
	pub fn tex_dimensions_vec2(&self) -> glam::Vec2
	{
		glam::UVec2::from(self.tex_dimensions()).as_vec2()
	}
}
impl super::UIElement for Img
{
	fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), DrawError>
	{
		self.quad.draw(render_ctx)
	}
}
