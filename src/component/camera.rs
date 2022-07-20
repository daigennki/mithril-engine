/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use glam::*;
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::buffer::BufferUsage;
use crate::render::RenderContext;

pub struct Camera
{
	descriptor_set: Arc<PersistentDescriptorSet>,
	projview: Mat4
}
impl Camera
{
	pub fn new(render_ctx: &mut RenderContext, pos: Vec3, target: Vec3) -> Result<Camera, Box<dyn std::error::Error>>
	{
		// Create a camera facing `target` from `pos` with 1 radians vertical FOV.
		// TODO: use actual window aspect ratio rather than a constant
		// TODO: adjust near/far values to be more sensible
		let proj = Mat4::perspective_lh(1.0, 16.0 / 9.0, 0.01, 1000.0);
		let view = Mat4::look_at_lh(pos, target, Vec3::Z);
		let projview = proj * view;
		let projview_buf = render_ctx.new_buffer(projview.to_cols_array(), BufferUsage::uniform_buffer())?;

		Ok(Camera{
			descriptor_set: render_ctx.new_descriptor_set("World", 1, [
				WriteDescriptorSet::buffer(0, projview_buf.clone())
			])?,
			projview: projview
		})
	}

	/// Bind this camera's projection and view matrices so they can be used in shaders.
	pub fn bind(&self, render_ctx: &mut RenderContext) -> Result<(), crate::render::PipelineNotLoaded>
	{
		// this must be bound as descriptor set 1
		render_ctx.bind_descriptor_set(1, self.descriptor_set.clone())
	}
}

