/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::buffer::ImmutableBuffer;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use super::rendercontext::texture::Texture;
use super::rendercontext::RenderContext;

pub struct Image
{
	tex: Texture,
	vertex_buf: Arc<ImmutableBuffer<[f32; 8]>>,
	transform_buf: Arc<ImmutableBuffer<glam::Mat4>>,
	descriptor_set: Arc<PersistentDescriptorSet>
}
impl Image
{
	pub fn new(render_ctx: &mut RenderContext, path: &std::path::Path) 
		-> Result<Image, Box<dyn std::error::Error>>
	{
		let transformation = glam::Mat4::IDENTITY;
		let transform_buf = render_ctx.new_buffer(transformation, BufferUsage::uniform_buffer())?;

		// texture
		let tex = render_ctx.new_texture(path)?;

		// vertex data (common for both position and UV)
		let vertices: [f32; 8] = [
			0.0, 0.0,
			1.0, 0.0,
			0.0, 1.0,
			1.0, 1.0
		];
		let vertex_buf = render_ctx.new_buffer(vertices, BufferUsage::vertex_buffer())?;

		let set_layout = render_ctx.get_ui_set_layout();

		// create descriptor set
		let descriptor_set = PersistentDescriptorSet::new(set_layout, [
			WriteDescriptorSet::buffer(0, transform_buf.clone()),
			WriteDescriptorSet::image_view(1, tex.clone_view())
		])?;

		Ok(Image{
			tex: render_ctx.new_texture(path)?,
			transform_buf: transform_buf,
			descriptor_set: descriptor_set,
			vertex_buf: vertex_buf
		})
	}

	pub fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		render_ctx.bind_ui_descriptor_set(0, self.descriptor_set.clone())?;
		render_ctx.bind_vertex_buffers(0, self.vertex_buf.clone())?;
		render_ctx.bind_vertex_buffers(1, self.vertex_buf.clone())?;
		render_ctx.draw(4, 1, 0, 0)?;
		Ok(())
	}
}
