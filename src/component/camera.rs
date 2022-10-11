/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use glam::*;
use bytemuck::{ Pod, Zeroable };
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::buffer::{ BufferUsage, cpu_access::CpuAccessibleBuffer };
use serde::Deserialize;
use crate::render::{ RenderContext, command_buffer::CommandBuffer };
use crate::component::{ UniqueComponent, DeferGpuResourceLoading };
use crate::GenericEngineError;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct CameraData
{
	projview: Mat4,
	proj: Mat4,
	view: Mat4
}

#[derive(shipyard::Unique, Deserialize, UniqueComponent)]
pub struct Camera
{
	#[serde(skip)]
	projview_buf: Option<Arc<CpuAccessibleBuffer<CameraData>>>,
	#[serde(skip)]
	descriptor_set: Option<Arc<PersistentDescriptorSet>>,

	position: Vec3,
	target: Vec3,

	#[serde(skip)]
	width: u32,
	#[serde(skip)]
	height: u32
}
impl Camera
{
	/*pub fn new(render_ctx: &mut RenderContext, pos: Vec3, target: Vec3) -> Result<Camera, GenericEngineError>
	{
		let dim = render_ctx.swapchain_dimensions();
		let projview = calculate_projview(pos, target, dim[0], dim[1]);
		let projview_buf = render_ctx.new_cpu_buffer_from_data(projview, BufferUsage::uniform_buffer())?;

		Ok(Camera{
			projview_buf: projview_buf.clone(),
			descriptor_set: render_ctx.new_descriptor_set("World", 1, [
				WriteDescriptorSet::buffer(0, projview_buf)
			])?,
		})
	}*/

	pub fn update_window_size(&mut self, width: u32, height: u32) -> Result<(), GenericEngineError>
	{
		self.width = width;
		self.height = height;
		*self.projview_buf.as_ref().ok_or("camera not loaded")?.write()? = 
			calculate_projview(self.position, self.target, width, height);
		Ok(())
	}

	pub fn set_pos_and_target(&mut self, pos: Vec3, target: Vec3) -> Result<(), GenericEngineError>
	{
		self.position = pos;
		self.target = target;
		*self.projview_buf.as_ref().ok_or("camera not loaded")?.write()? = 
			calculate_projview(pos, target, self.width, self.height);
		Ok(())
	}

	/// Bind this camera's projection and view matrices so they can be used in shaders.
	pub fn bind<L>(&self, cb: &mut CommandBuffer<L>) -> Result<(), GenericEngineError>
	{
		// this must be bound as descriptor set 1
		cb.bind_descriptor_set(1, self.descriptor_set.as_ref().ok_or("camera not loaded")?.clone())?;
		Ok(())
	}
}
impl DeferGpuResourceLoading for Camera
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		let dim = render_ctx.swapchain_dimensions();
		let projview = calculate_projview(self.position, self.target, dim[0], dim[1]);
		let projview_buf = render_ctx.new_cpu_buffer_from_data(projview, BufferUsage{ uniform_buffer: true, ..BufferUsage::empty() })?;
		self.descriptor_set = Some(render_ctx.new_descriptor_set("PBR", 1, [
			WriteDescriptorSet::buffer(0, projview_buf.clone())
		])?);
		self.projview_buf = Some(projview_buf);
		Ok(())
	}
}

/// Calculate the projection and view matrices from the given camera parameters.
/// Returns combined data to be given to the UBO.
fn calculate_projview(pos: Vec3, target: Vec3, width: u32, height: u32) -> CameraData
{
	// Create a camera facing `target` from `pos` with 1 radians vertical FOV.
	let aspect_ratio = width as f32 / height as f32;
	let proj = Mat4::perspective_lh(1.0, aspect_ratio, 0.01, 1000.0);
	let view = Mat4::look_at_lh(pos, target, Vec3::NEG_Z);

	CameraData{ 
		projview: proj * view, 
		proj: proj,
		view: view 
	}
}

