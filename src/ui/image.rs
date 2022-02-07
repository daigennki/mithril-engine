/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use super::rendercontext::texture::Texture;
use super::rendercontext::RenderContext;

pub struct Image
{
	tex: Texture
}
impl Image
{
	pub fn new(render_ctx: &mut RenderContext, path: &std::path::Path) 
		-> Result<Image, Box<dyn std::error::Error>>
	{
		Ok(Image{
			tex: render_ctx.new_texture(path)?
		})
	}
}
