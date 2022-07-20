/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod ui;
pub mod mesh;
pub mod camera;

use std::sync::Arc;
use glam::*;
use vulkano::buffer::{ ImmutableBuffer, BufferUsage };
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::render::RenderContext;

pub struct Transform
{
	// TODO: parent-child relationship
	buf: Arc<ImmutableBuffer<[f32]>>,
	descriptor_set: Arc<PersistentDescriptorSet>,
	pos: Vec3,
	scale: Vec3
}
impl Transform
{
	pub fn new(render_ctx: &mut RenderContext, pos: Vec3, scale: Vec3) -> Result<Transform, Box<dyn std::error::Error>>
	{
		let transform_mat = Mat4::from_scale_rotation_translation(
			scale,
			Quat::IDENTITY,
			pos
		);
		let buf = render_ctx.new_buffer(transform_mat.to_cols_array(), BufferUsage::uniform_buffer())?;
		
		Ok(Transform{ 
			buf: buf.clone(),
			descriptor_set: render_ctx.new_descriptor_set("World", 0, [
				WriteDescriptorSet::buffer(0, buf)
			])?, 
			pos: pos, 
			scale: scale 
		})
	}

	/*pub fn set_pos(&mut self, pos: Vec3)
	{
		self.pos = pos;
		let transform_mat = Mat4::from_scale_rotation_translation(
			scale,
			Quat::IDENTITY,
			pos
		);

		// TODO: this is probably inefficient, so use coherent memory so we don't have to re-create a buffer
		let buf = render_ctx.new_buffer(transform_mat.to_cols_array(), BufferUsage::uniform_buffer())?;
		self.
	}*/

	pub fn bind_descriptor_set(&self, render_ctx: &mut RenderContext) -> Result<(), crate::render::PipelineNotLoaded>
	{
		render_ctx.bind_descriptor_set(0, self.descriptor_set.clone())
	}
}

/// Convenience function: create a tuple of `Transform` and `Mesh` to display a simple triangle.
pub fn new_triangle(render_ctx: &mut RenderContext, pos: Vec3, scale: Vec3, color: Vec4)
	-> Result<(Transform, mesh::Mesh), Box<dyn std::error::Error>>
{
	let tri_transform = Transform::new(render_ctx, pos, scale)?;
	let tri_mesh = mesh::Mesh::new(render_ctx, color)?;

	Ok((tri_transform, tri_mesh))
}

