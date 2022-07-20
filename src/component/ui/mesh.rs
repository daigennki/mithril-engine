/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::buffer::{ ImmutableBuffer, BufferUsage };
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::command_buffer::DrawError;
use glam::*;
use crate::vertex::*;
use crate::render::texture::Texture;
use crate::render::RenderContext;

/// UI component that renders to a mesh, such as a quad, or a background frame mesh.
pub struct Mesh
{
	pos_vert_buf: Arc<ImmutableBuffer<[Vertex2]>>,
	uv_vert_buf: Arc<ImmutableBuffer<[Vertex2]>>,
	descriptor_set: Arc<PersistentDescriptorSet>,
}
impl Mesh
{
	pub fn new(render_ctx: &mut RenderContext, tex: Texture) -> Result<Mesh, Box<dyn std::error::Error>>
	{
		// vertex data
		let mut pos_verts = [
			Vertex2::new(0.0, 0.0),
			Vertex2::new(1.0, 0.0),
			Vertex2::new(0.0, 1.0),
			Vertex2::new(1.0, 1.0)
		];
		let uv_verts = pos_verts;

		// resize position vertices according to texture dimensions
		let dimensions_uvec2: UVec2 = tex.dimensions().width_height().into();
		let dimensions = dimensions_uvec2.as_vec2();
		let half_dimensions = dimensions / 2.0;
		for pos in &mut pos_verts {
			let pos_clone = pos.clone();
			let x = pos_clone.x * dimensions.x - half_dimensions.x;
			let y = pos_clone.y * dimensions.y - half_dimensions.y;
			*pos = Vertex2::new(x, y);
		}

		Ok(Mesh{
			descriptor_set: render_ctx.new_descriptor_set("UI", 1, [ WriteDescriptorSet::image_view(0, tex.view()) ])?,
			pos_vert_buf: render_ctx.new_buffer(pos_verts, BufferUsage::vertex_buffer())?,
			uv_vert_buf: render_ctx.new_buffer(uv_verts, BufferUsage::vertex_buffer())?,
		})
	}

	pub fn new_from_corners(render_ctx: &mut RenderContext, top_left: Vec2, bottom_right: Vec2, tex: Texture)
		-> Result<Mesh, Box<dyn std::error::Error>>
	{
		// vertex data
		let pos_verts: [Vertex2; 4] = [
			Vertex2::new_from_vec2(top_left),
			Vertex2::new(bottom_right.x, top_left.y),
			Vertex2::new(top_left.x, bottom_right.y),
			Vertex2::new_from_vec2(bottom_right)
		];
		let uv_verts = [
			Vertex2::new(0.0, 0.0),
			Vertex2::new(1.0, 0.0),
			Vertex2::new(0.0, 1.0),
			Vertex2::new(1.0, 1.0)
		];

		Ok(Mesh{
			descriptor_set: render_ctx.new_descriptor_set("UI", 1, [ WriteDescriptorSet::image_view(0, tex.view()) ])?,
			pos_vert_buf: render_ctx.new_buffer(pos_verts, BufferUsage::vertex_buffer())?,
			uv_vert_buf: render_ctx.new_buffer(uv_verts, BufferUsage::vertex_buffer())?,
		})
	}

	pub fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		render_ctx.bind_descriptor_set("UI", 1, self.descriptor_set.clone())?;
		render_ctx.bind_vertex_buffers(0, (self.pos_vert_buf.clone(), self.uv_vert_buf.clone()));
		render_ctx.draw(4, 1, 0, 0)?;
		Ok(())
	}
}
