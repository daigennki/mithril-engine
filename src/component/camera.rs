/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use glam::*;
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::buffer::{ BufferUsage, cpu_access::CpuAccessibleBuffer };
use crate::render::RenderContext;

pub struct Camera
{
	projview_buf: Arc<CpuAccessibleBuffer<[f32]>>,
	descriptor_set: Arc<PersistentDescriptorSet>,
	projview: Mat4
}
impl Camera
{
	pub fn new(render_ctx: &mut RenderContext, pos: Vec3, target: Vec3) -> Result<Camera, Box<dyn std::error::Error>>
	{
		let dim = render_ctx.swapchain_dimensions();
		let projview = calculate_projview(pos, target, dim[0] as f32, dim[1] as f32);
		let projview_buf = render_ctx.new_cpu_buffer(projview.to_cols_array(), BufferUsage::uniform_buffer())?;

		Ok(Camera{
			projview_buf: projview_buf.clone(),
			descriptor_set: render_ctx.new_descriptor_set("World", 1, [
				WriteDescriptorSet::buffer(0, projview_buf)
			])?,
			projview: projview
		})
	}

	pub fn set_pos_and_target(&mut self, render_ctx: &mut RenderContext, pos: Vec3, target: Vec3)
		-> Result<(), Box<dyn std::error::Error>>
	{
		let dim = render_ctx.swapchain_dimensions();
		self.projview = calculate_projview(pos, target, dim[0] as f32, dim[1] as f32);
		self.projview_buf.write()?.clone_from_slice(&self.projview.to_cols_array());
		Ok(())
	}

	/// Bind this camera's projection and view matrices so they can be used in shaders.
	pub fn bind(&self, render_ctx: &mut RenderContext) -> Result<(), crate::render::PipelineNotLoaded>
	{
		// this must be bound as descriptor set 1
		render_ctx.bind_descriptor_set(1, self.descriptor_set.clone())
	}
}

fn calculate_projview(pos: Vec3, target: Vec3, width: f32, height: f32) -> Mat4
{
	// Create a camera facing `target` from `pos` with 1 radians vertical FOV.
	let aspect_ratio = width / height;
	let proj = Mat4::perspective_lh(1.0, aspect_ratio, 0.01, 1000.0);
	let view = Mat4::look_at_lh(pos, target, Vec3::Z);
	proj * view
}

