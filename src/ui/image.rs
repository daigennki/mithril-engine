/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use super::rendercontext::RenderContext;
use super::quad::Quad;

pub struct Image
{
	quad: Quad
}
impl Image
{
	pub fn new(render_ctx: &mut RenderContext, pos: glam::Vec2, proj: glam::Mat4, path: &std::path::Path) 
		-> Result<Image, Box<dyn std::error::Error>>
	{
		let tex = render_ctx.new_texture(path)?;
		let dim = tex.dimensions();
		let scale = glam::Vec2::new(dim.width() as f32, dim.height() as f32);
		let quad = Quad::new(render_ctx, pos, scale, proj, tex)?;

		Ok(Image{
			quad: quad
		})
	}
}
impl super::UIElement for Image
{
	fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		self.quad.draw(render_ctx)
	}
}
