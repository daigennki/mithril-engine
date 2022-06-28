/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use glam::*;
use vulkano::buffer::{ ImmutableBuffer, BufferUsage };
use vulkano::command_buffer::DrawError;
use crate::vertex::*;
use crate::render::texture::Texture;
use crate::render::RenderContext;

pub struct Mesh
{
	pos_vert_buf: Arc<ImmutableBuffer<[Vertex3]>>,
	uv_vert_buf: Arc<ImmutableBuffer<[Vertex2]>>,
	index_buf: Arc<ImmutableBuffer<[u32]>>
}
impl Mesh
{
	pub fn new(render_ctx: &mut RenderContext) -> Result<Mesh, Box<dyn std::error::Error>>
	{
		let pos_verts = vec![ 
			Vertex3::new(0.0, 0.0, 0.0),
			Vertex3::new(2.0, 0.0, 0.0),
			Vertex3::new(1.0, 1.0, 0.0)
		];

		let uv_verts = vec![ 
			Vertex2::new(0.0, 0.0),
			Vertex2::new(1.0, 0.0),
			Vertex2::new(0.5, 1.0)
		];

		let indices = vec![
			0, 1, 2
		];

		Ok(Mesh{
			pos_vert_buf: render_ctx.new_buffer(pos_verts, BufferUsage::vertex_buffer())?,
			uv_vert_buf: render_ctx.new_buffer(uv_verts, BufferUsage::vertex_buffer())?,
			index_buf: render_ctx.new_buffer(indices, BufferUsage::index_buffer())?
		})
	}

	pub fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), DrawError>
	{
		//render_ctx.bind_ui_descriptor_set(1, self.descriptor_set.clone());
		render_ctx.bind_vertex_buffers(0, (self.pos_vert_buf.clone(), self.uv_vert_buf.clone()));
		render_ctx.bind_index_buffers(self.index_buf.clone());
		render_ctx.draw(4, 1, 0, 0)?;
		Ok(())
	}
}