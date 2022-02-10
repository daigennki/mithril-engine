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

/// A base UI element that renders to a quad.
pub(super) struct Quad
{
	vertex_buf: Arc<ImmutableBuffer<[f32; 8]>>,
	descriptor_set: Arc<PersistentDescriptorSet>
}
impl Quad
{
	pub fn new(render_ctx: &mut RenderContext, pos: glam::Vec2, scale: glam::Vec2, proj: glam::Mat4, tex: Texture) 
		-> Result<Quad, Box<dyn std::error::Error>>
	{
		let transformation = proj * glam::Mat4::from_scale_rotation_translation(
			glam::Vec3::new(scale.x, scale.y, 1.0), 
			glam::Quat::IDENTITY, 
			glam::Vec3::new(pos.x, pos.y, 0.0)
		);
		let transform_buf = render_ctx.new_buffer(transformation, BufferUsage::uniform_buffer())?;

		// create descriptor set
		let set_layout = render_ctx.get_ui_set_layout();
		let descriptor_set = PersistentDescriptorSet::new(set_layout, [
			WriteDescriptorSet::buffer(0, transform_buf.clone()),
			WriteDescriptorSet::image_view(1, tex.clone_view())
		])?;

		// vertex data (common for both position and UV)
		let vertices: [f32; 8] = [
			0.0, 0.0,
			1.0, 0.0,
			0.0, 1.0,
			1.0, 1.0
		];
		let vertex_buf = render_ctx.new_buffer(vertices, BufferUsage::vertex_buffer())?;

		Ok(Quad{
			descriptor_set: descriptor_set,
			vertex_buf: vertex_buf
		})
	}
}
impl super::UIElement for Quad
{
	fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		render_ctx.bind_ui_descriptor_set(0, self.descriptor_set.clone())?;
		render_ctx.bind_vertex_buffers(0, self.vertex_buf.clone())?;
		render_ctx.bind_vertex_buffers(1, self.vertex_buf.clone())?;
		render_ctx.draw(4, 1, 0, 0)?;
		Ok(())
	}
}
