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
use crate::render::texture::Texture;
use crate::render::RenderContext;

/// UI component that renders to a mesh, such as a quad, or a background frame mesh.
pub struct Mesh
{
	pos_vert_buf: Arc<ImmutableBuffer<[Vec2]>>,
	uv_vert_buf: Arc<ImmutableBuffer<[Vec2]>>,
	descriptor_set: Arc<PersistentDescriptorSet>,
}
impl Mesh
{
	pub fn new(render_ctx: &mut RenderContext, tex: Texture) -> Result<Mesh, Box<dyn std::error::Error>>
	{
		// vertex data
		let mut pos_verts: [Vec2; 4] = [
			Vec2::new(0.0, 0.0),
			Vec2::new(1.0, 0.0),
			Vec2::new(0.0, 1.0),
			Vec2::new(1.0, 1.0)
		];
		let uv_verts = pos_verts;

		// resize position vertices according to texture dimensions
		let dimensions_uvec2: UVec2 = tex.dimensions().width_height().into();
		let dimensions = dimensions_uvec2.as_vec2();
		let half_dimensions = dimensions / 2.0;
		for pos in &mut pos_verts {
			*pos = pos.clone() * dimensions - half_dimensions;
		}

		Ok(Mesh{
			descriptor_set: render_ctx.new_ui_descriptor_set(1, [ WriteDescriptorSet::image_view(0, tex.view()) ])?,
			pos_vert_buf: render_ctx.new_buffer(pos_verts, BufferUsage::vertex_buffer())?,
			uv_vert_buf: render_ctx.new_buffer(uv_verts, BufferUsage::vertex_buffer())?,
		})
	}

	pub fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), DrawError>
	{
		render_ctx.bind_ui_descriptor_set(1, self.descriptor_set.clone());
		render_ctx.bind_vertex_buffers(0, (self.pos_vert_buf.clone(), self.uv_vert_buf.clone()));
		render_ctx.draw(4, 1, 0, 0)?;
		Ok(())
	}
}
