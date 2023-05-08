/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use glam::*;
use serde::Deserialize;
use shipyard::EntityId;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};

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

#[derive(shipyard::Unique)]
pub struct CameraManager
{
	active_camera: EntityId,
	default_fov: CameraFov,
	projview: Mat4,
	sky_projview: Mat4,
}
impl CameraManager
{
	pub fn new(render_ctx: &mut RenderContext, default_fov: CameraFov) -> Result<Self, GenericEngineError>
	{
		let dim = render_ctx.swapchain_dimensions();
		let (projview, sky_projview) = calculate_projview(Vec3::ZERO, Vec3::Y, dim[0], dim[1], default_fov);

		Ok(CameraManager {
			active_camera: Default::default(),
			default_fov,
			projview,
			sky_projview,
		})
	}

	/// This function *must* be run every frame, before entities are rendered.
	pub fn update(
		&mut self,
		render_ctx: &mut RenderContext,
		current_pos: Vec3,
		current_rotation: &Quat,
		fov: Option<CameraFov>,
	) -> Result<(), GenericEngineError>
	{
		let target = *current_rotation * Vec3::Y;
		let dim = render_ctx.swapchain_dimensions();
		let (projview, sky_projview) = calculate_projview(current_pos, target, dim[0], dim[1], fov.unwrap_or(self.default_fov));
		self.projview = projview;
		self.sky_projview = sky_projview;
		Ok(())
	}

	/// Push the multiplied projection and view matrix so they can be used in shaders.
	pub fn push_projview(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>)
		-> Result<(), GenericEngineError>
	{
		crate::render::push_constants(cb, 0, self.projview)?;
		Ok(())
	}

	/// Push the multiplied projection and view matrix so they can be used specifically in the skybox shader.
	/// (the view matrix here never changes its eye position from (0, 0, 0))
	pub fn push_sky_projview(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
	) -> Result<(), GenericEngineError>
	{
		crate::render::push_constants(cb, 0, self.sky_projview)?;
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

/// Calculate the projection and view matrices from the given camera parameters.
/// Returns a pair of the typical projview and the skybox projview.
fn calculate_projview(pos: Vec3, dir: Vec3, width: u32, height: u32, fov: CameraFov) -> (Mat4, Mat4)
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

	let projview = proj * view;
	let sky_projview = proj * sky_view;

	(projview, sky_projview)
}
