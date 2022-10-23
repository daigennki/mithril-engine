/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use glam::*;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;

use crate::render::{texture::Texture, RenderContext};
use crate::GenericEngineError;

/// UI component that renders to a mesh, such as a quad, or a background frame mesh.
#[derive(shipyard::Component)]
pub struct Mesh
{
	pos_vert_buf: Arc<DeviceLocalBuffer<[Vec2]>>,
	uv_vert_buf: Arc<DeviceLocalBuffer<[Vec2]>>,
	descriptor_set: Arc<PersistentDescriptorSet>,
}
impl Mesh
{
	pub fn new(render_ctx: &mut RenderContext, tex: Texture) -> Result<Self, GenericEngineError>
	{
		// resize position vertices according to texture dimensions
		let dimensions = UVec2::from_array(tex.dimensions().width_height()).as_vec2();
		let half_dimensions = dimensions * 0.5;
		Self::new_from_corners(render_ctx, -half_dimensions, half_dimensions, tex)
	}

	pub fn new_from_corners(
		render_ctx: &mut RenderContext, top_left: Vec2, bottom_right: Vec2, tex: Texture,
	) -> Result<Self, GenericEngineError>
	{
		// vertex data
		let pos_verts = [
			top_left,
			Vec2::new(top_left.x, bottom_right.y),
			Vec2::new(bottom_right.x, top_left.y),
			bottom_right,
		];
		let uv_verts = [
			Vec2::new(0.0, 0.0),
			Vec2::new(0.0, 1.0),
			Vec2::new(1.0, 0.0),
			Vec2::new(1.0, 1.0),
		];

		let vbo_usage = BufferUsage { vertex_buffer: true, ..BufferUsage::empty() };
		Ok(Mesh {
			descriptor_set: render_ctx.new_descriptor_set("UI", 1, [WriteDescriptorSet::image_view(0, tex.view())])?,
			pos_vert_buf: render_ctx.new_buffer_from_iter(pos_verts, vbo_usage)?,
			uv_vert_buf: render_ctx.new_buffer_from_iter(uv_verts, vbo_usage)?,
		})
	}

	pub fn draw(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		crate::render::bind_descriptor_set(cb, 1, self.descriptor_set.clone())?;
		cb.bind_vertex_buffers(0, (self.pos_vert_buf.clone(), self.uv_vert_buf.clone()));
		cb.draw(4, 1, 0, 0)?;
		Ok(())
	}
}
