/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::buffer::ImmutableBuffer;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::command_buffer::DrawError;
use crate::rendercontext::texture::Texture;
use crate::rendercontext::RenderContext;

/// A base UI element that renders to a quad.
pub(super) struct Quad
{
	vertex_buf: Arc<ImmutableBuffer<[f32; 8]>>,
	descriptor_set: Arc<PersistentDescriptorSet>,
	tex: Texture
}
impl Quad
{
	pub fn new(render_ctx: &mut RenderContext, tex: Texture) 
		-> Result<Quad, Box<dyn std::error::Error>>
	{
		// create descriptor set
		let set_layout = render_ctx.get_ui_set_layout(1);
		let descriptor_set = PersistentDescriptorSet::new(set_layout, [
			WriteDescriptorSet::image_view(0, tex.clone_view())
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
			vertex_buf: vertex_buf,
			tex: tex
		})
	}

	pub fn tex_dimensions(&self) -> [u32; 2]
	{
		let dim = self.tex.dimensions();
		[ dim.width(), dim.height() ]
	}
}
impl super::UIElement for Quad
{
	fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), DrawError>
	{
		render_ctx.bind_ui_descriptor_set(1, self.descriptor_set.clone());
		render_ctx.bind_vertex_buffers(0, self.vertex_buf.clone());
		render_ctx.bind_vertex_buffers(1, self.vertex_buf.clone());
		render_ctx.draw(4, 1, 0, 0)?;
		Ok(())
	}
}
