/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod ui;
pub mod mesh;
pub mod camera;

use std::sync::Arc;
use glam::*;
use vulkano::buffer::{ /*ImmutableBuffer,*/ BufferUsage, cpu_access::CpuAccessibleBuffer };
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::render::RenderContext;

pub struct Transform
{
	// TODO: parent-child relationship
	// TODO: maybe we should use immutable buffers but only for static objects...
	buf: Arc<CpuAccessibleBuffer<[f32]>>,
	descriptor_set: Arc<PersistentDescriptorSet>,
	position: Vec3,
	scale: Vec3,
	rot_quat: Quat
}
impl Transform
{
	pub fn new(render_ctx: &mut RenderContext, position: Vec3, scale: Vec3, rotation: Vec3) -> Result<Transform, Box<dyn std::error::Error>>
	{
		let rot_quat = Quat::from_euler(EulerRot::XYZ, rotation.x, rotation.y, rotation.z);
		let transform_mat = Mat4::from_scale_rotation_translation(
			scale,
			rot_quat,
			position
		);
		let buf = render_ctx.new_cpu_buffer(transform_mat.to_cols_array(), BufferUsage::uniform_buffer())?;

		Ok(Transform{ 
			buf: buf.clone(),
			descriptor_set: render_ctx.new_descriptor_set("World", 0, [
				WriteDescriptorSet::buffer(0, buf)
			])?, 
			position: position, 
			scale: scale,
			rot_quat: rot_quat
		})
	}
	

	fn update_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		let transform_mat = Mat4::from_scale_rotation_translation(self.scale, self.rot_quat, self.position);
		self.buf.write()?.clone_from_slice(&transform_mat.to_cols_array());
		Ok(())
	}

	pub fn set_pos(&mut self, position: Vec3) -> Result<(), Box<dyn std::error::Error>>
	{
		self.position = position;
		self.update_buffer()
	}
	
	pub fn set_scale(&mut self, scale: Vec3) -> Result<(), Box<dyn std::error::Error>>
	{
		self.scale = scale;
		self.update_buffer()
	}

	/// Set the rotation of this object, in terms of X, Y, and Z axis rotations.
	pub fn set_rotation(&mut self, rotation: Vec3) -> Result<(), Box<dyn std::error::Error>>
	{
		self.rot_quat = Quat::from_euler(EulerRot::XYZ, rotation.x, rotation.y, rotation.z);
		self.update_buffer()
	}

	pub fn bind_descriptor_set(&self, render_ctx: &mut RenderContext) -> Result<(), crate::render::PipelineNotLoaded>
	{
		render_ctx.bind_descriptor_set(0, self.descriptor_set.clone())
	}
}

/// Convenience function: create a tuple of `Transform` and `Mesh` to display a simple triangle.
pub fn new_triangle(render_ctx: &mut RenderContext, pos: Vec3, scale: Vec3, rot: Vec3, color: Vec4)
	-> Result<(Transform, mesh::Mesh), Box<dyn std::error::Error>>
{
	let tri_transform = Transform::new(render_ctx, pos, scale, rot)?;
	let tri_mesh = mesh::Mesh::new(render_ctx, color)?;

	Ok((tri_transform, tri_mesh))
}

