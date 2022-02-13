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

/// UI component that renders to a mesh, such as a quad, or a background frame mesh.
pub struct Quad
{
	pos_vert_buf: Arc<ImmutableBuffer<[glam::Vec2]>>,
	uv_vert_buf: Arc<ImmutableBuffer<[glam::Vec2]>>,
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

		// vertex data
		let mut pos_verts: [glam::Vec2; 4] = [
			[ 0.0, 0.0 ].into(),
			[ 1.0, 0.0 ].into(),
			[ 0.0, 1.0 ].into(),
			[ 1.0, 1.0 ].into()
		];
		let uv_verts = pos_verts;

		// resize position vertices according to texture dimensions
		let tex_dimensions = tex.dimensions();
		let width = tex_dimensions.width() as f32;
		let height = tex_dimensions.height() as f32;
		let half_width = width / 2.0;
		let half_height = height / 2.0;
		for pos in &mut pos_verts {
			pos.x = pos.x * width - half_width;
			pos.y = pos.y * height - half_height;
		}

		// create vertex buffers
		let pos_vert_buf = render_ctx.new_buffer(pos_verts, BufferUsage::vertex_buffer())?;
		let uv_vert_buf = render_ctx.new_buffer(uv_verts, BufferUsage::vertex_buffer())?;

		Ok(Quad{
			descriptor_set: descriptor_set,
			pos_vert_buf: pos_vert_buf,
			uv_vert_buf: uv_vert_buf,
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
		render_ctx.bind_vertex_buffers(0, self.pos_vert_buf.clone());
		render_ctx.bind_vertex_buffers(1, self.uv_vert_buf.clone());
		render_ctx.draw(4, 1, 0, 0)?;
		Ok(())
	}
}
