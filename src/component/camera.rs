/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use bytemuck::{Pod, Zeroable};
use glam::*;
use serde::Deserialize;
use shipyard::EntityId;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuBufferPool, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;

use crate::component::EntityComponent;
use crate::render::RenderContext;
use crate::GenericEngineError;

/// Enum for the camera FoV (Field of View), in degrees.
/// The FoV for the other axis is calculated automatically from the current window aspect ratio.
#[derive(Copy, Clone, Deserialize)]
pub enum CameraFov
{
	/// Horizontal FoV
	X(f32),

	/// Vertical FoV
	Y(f32),
}

#[derive(shipyard::Unique)]
pub struct CameraManager
{
	staging_buf: CpuBufferPool<CameraData>,
	projview_buf: Arc<DeviceLocalBuffer<CameraData>>,
	descriptor_set: Arc<PersistentDescriptorSet>,

	active_camera: EntityId,

	default_fov: CameraFov,

	projview: Mat4,
}
impl CameraManager
{
	pub fn new(render_ctx: &mut RenderContext, default_fov: CameraFov) -> Result<Self, GenericEngineError>
	{
		let dim = render_ctx.swapchain_dimensions();
		let projview_struct = calculate_projview(Vec3::ZERO, Vec3::Y, dim[0], dim[1], default_fov);
		let projview = projview_struct.projview;
		let (staging_buf, projview_buf) = render_ctx
			.new_cpu_buffer_from_data(projview_struct, BufferUsage { uniform_buffer: true, ..BufferUsage::empty() })?;

		let descriptor_set = render_ctx.new_descriptor_set("PBR", 1, [WriteDescriptorSet::buffer(
			0,
			projview_buf.clone(),
		)])?;

		Ok(CameraManager{
			staging_buf,
			projview_buf,
			descriptor_set,
			active_camera: Default::default(),
			default_fov,
			projview,
		})
	}

	/// This function *must* be run every frame, before entities are rendered.
	/// It updates the GPU buffer which contains the projection and view matrices.
	pub fn update(
		&mut self, render_ctx: &mut RenderContext, current_pos: Vec3, current_rotation: &Quat, fov: Option<CameraFov>
	) -> Result<(), GenericEngineError>
	{
		let target = *current_rotation * Vec3::Y;
		let dim = render_ctx.swapchain_dimensions();
		let projview_struct = calculate_projview(current_pos, target, dim[0], dim[1], fov.unwrap_or(self.default_fov));
		self.projview = projview_struct.projview;

		let staged = self.staging_buf.from_data(projview_struct)?;
		render_ctx.copy_buffer(staged, self.projview_buf.clone())?;

		Ok(())
	}

	/// Bind the projection and view matrices so they can be used in shaders.
	pub fn bind(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		// this must be bound as descriptor set 1
		crate::render::bind_descriptor_set(cb, 1, self.descriptor_set.clone())?;
		Ok(())
	}

	pub fn set_active(&mut self, eid: EntityId)
	{
		self.active_camera = eid
	}
	pub fn active_camera(&self) -> EntityId
	{
		self.active_camera
	}

	pub fn projview(&self) -> Mat4
	{
		self.projview
	}
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct CameraData
{
	projview: Mat4,
	sky_projview: Mat4,
}

/// A component that turns an entity into a camera.
/// The entity must have a `Transform` component.
/// The `Transform` component's position and rotation will be used for the camera's position and rotation.
#[derive(shipyard::Component, Deserialize, EntityComponent)]
pub struct Camera
{
	/// This camera's field of view.
	/// If this is `None`, then the `CameraManager`'s default FoV will be used instead.
	pub fov: Option<CameraFov>,
}

/// Calculate the projection and view matrices from the given camera parameters.
/// Returns combined data to be given to the UBO.
fn calculate_projview(pos: Vec3, dir: Vec3, width: u32, height: u32, fov: CameraFov) -> CameraData
{
	let aspect_ratio = width as f32 / height as f32;
	let fov_y_deg = match fov {
		CameraFov::X(fov_x_deg) => fov_x_deg / aspect_ratio,
		CameraFov::Y(fov_y_deg) => fov_y_deg,
	};
	let fov_y_rad = fov_y_deg * std::f32::consts::PI / 180.0;

	let proj = Mat4::perspective_lh(fov_y_rad, aspect_ratio, 0.25, 5000.0);
	let view = Mat4::look_at_lh(pos, pos + dir, Vec3::NEG_Z);
	let sky_view = Mat4::look_at_lh(Vec3::ZERO, dir, Vec3::NEG_Z);

	CameraData { 
		projview: proj * view, 
		sky_projview: proj * sky_view,
	}
}
