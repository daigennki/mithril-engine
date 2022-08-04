/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use glam::*;
use vulkano::buffer::{ ImmutableBuffer, BufferUsage };
use vulkano::descriptor_set::{ WriteDescriptorSet, PersistentDescriptorSet };
use crate::render::RenderContext;

pub struct Mesh
{
	pos_vert_buf: Arc<ImmutableBuffer<[Vec3]>>,
	uv_vert_buf: Arc<ImmutableBuffer<[Vec2]>>,
	index_buf: Arc<ImmutableBuffer<[u32]>>,

	mat_set: Arc<PersistentDescriptorSet>
}
impl Mesh
{
	// TODO: set material
	pub fn new(render_ctx: &mut RenderContext, color: Vec4) -> Result<Mesh, Box<dyn std::error::Error>>
	{
		let pos_verts = vec![ 
			Vec3::new(-1.0, -0.5, 0.0),
			Vec3::new(1.0, -0.5, 0.0),
			Vec3::new(0.0, 0.5, 0.0)
		];

		let uv_verts = vec![ 
			Vec2::new(0.0, 0.0),
			Vec2::new(1.0, 0.0),
			Vec2::new(0.5, 1.0)
		];

		let indices = vec![
			0, 1, 2
		];

		let mat_buf = render_ctx.new_buffer_from_data(color, BufferUsage::uniform_buffer())?;
		let mat_set = render_ctx.new_descriptor_set("World", 2, [
			WriteDescriptorSet::buffer(0, mat_buf)
		])?;

		Ok(Mesh{
			pos_vert_buf: render_ctx.new_buffer_from_iter(pos_verts, BufferUsage::vertex_buffer())?,
			uv_vert_buf: render_ctx.new_buffer_from_iter(uv_verts, BufferUsage::vertex_buffer())?,
			index_buf: render_ctx.new_buffer_from_iter(indices, BufferUsage::index_buffer())?,
			mat_set: mat_set
		})
	}

	pub fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		render_ctx.bind_descriptor_set(2, self.mat_set.clone())?;
		render_ctx.bind_vertex_buffers(0, (self.pos_vert_buf.clone(), self.uv_vert_buf.clone()));
		render_ctx.bind_index_buffers(self.index_buf.clone());
		render_ctx.draw(3, 1, 0, 0)?;
		Ok(())
	}
}
