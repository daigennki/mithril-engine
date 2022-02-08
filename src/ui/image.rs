/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::buffer::ImmutableBuffer;
use vulkano::buffer::BufferUsage;
use super::rendercontext::texture::Texture;
use super::rendercontext::RenderContext;

pub struct Image
{
	tex: Texture,
	transform_buf: Arc<ImmutableBuffer<glam::Mat4>>
}
impl Image
{
	pub fn new(render_ctx: &mut RenderContext, path: &std::path::Path) 
		-> Result<Image, Box<dyn std::error::Error>>
	{
		let transformation = glam::Mat4::IDENTITY;
		let buf_usage = BufferUsage {
			uniform_buffer: true,
			..BufferUsage::none()
		};
		let transform_buf = render_ctx.new_buffer(transformation, buf_usage)?;

		Ok(Image{
			tex: render_ctx.new_texture(path)?,
			transform_buf: transform_buf
		})
	}
}
