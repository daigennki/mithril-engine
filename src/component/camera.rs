/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use shipyard::{
	iter::{IntoIter, IntoWithId},
	EntityId, Get, IntoWorkloadSystem, UniqueView, UniqueViewMut, View, WorkloadSystem,
};

use crate::component::{EntityComponent, WantsSystemAdded};
use crate::render::RenderContext;

pub const CAMERA_NEAR: f64 = 0.25;
pub const CAMERA_FAR: f64 = 500.0;

/// Enum for the camera FoV (Field of View), in degrees.
/// The FoV for the other axis is calculated automatically from the current window aspect ratio.
#[derive(Copy, Clone, Deserialize)]
pub enum CameraFov
{
	/// Horizontal FoV
	X(f64),

	/// Vertical FoV
	Y(f64),
}

/// A component that turns an entity into a camera.
/// The entity must have a `Transform` component.
/// The `Transform` component's position and rotation will be used for the camera's position and rotation.
#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct Camera
{
	/// This camera's field of view.
	/// If this is `None`, then the `CameraManager`'s default FoV will be used instead.
	pub fov: Option<CameraFov>,
}
impl WantsSystemAdded for Camera
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		Some(select_default_camera.into_workload_system().unwrap())
	}
	fn add_prerender_system(&self) -> Option<WorkloadSystem>
	{
		Some(update_camera.into_workload_system().unwrap())
	}
}
fn select_default_camera(mut camera_manager: UniqueViewMut<CameraManager>, cameras: View<Camera>)
{
	// If an active camera is not set, set the first inserted camera as the active one.
	if camera_manager.active_camera() == EntityId::dead() {
		if let Some((eid, _)) = cameras.inserted().iter().with_id().next() {
			camera_manager.set_active(eid);
		}
	}
}
fn update_camera(
	render_ctx: UniqueView<RenderContext>,
	transforms: View<super::Transform>,
	mut camera_manager: UniqueViewMut<CameraManager>,
	cameras: View<Camera>,
)
{
	let active_camera_id = camera_manager.active_camera();
	if let Ok((t, cam)) = (&transforms, &cameras).get(active_camera_id) {
		camera_manager.update(render_ctx.swapchain_dimensions(), t.position, &t.rotation_quat(), cam.fov);
	}
}

#[derive(shipyard::Unique)]
pub struct CameraManager
{
	active_camera: EntityId,
	default_fov: CameraFov,

	view: DMat4,
	projview: DMat4,
	sky_projview: DMat4,
	current_fov_y_rad: f64,
	current_aspect_ratio: f64,
}
impl CameraManager
{
	pub fn new(viewport_extent: [u32; 2], default_fov: CameraFov) -> Self
	{
		let aspect_ratio = viewport_extent[0] as f64 / viewport_extent[1] as f64;
		let fov_y_rad = match default_fov {
			CameraFov::X(fov_x) => fov_x / aspect_ratio,
			CameraFov::Y(fov_y) => fov_y,
		}
		.to_radians();

		// the view matrix position is initially all zero, so it can be used to
		// initialize the sky view matrix as well
		let proj = DMat4::perspective_lh(fov_y_rad, aspect_ratio, CAMERA_NEAR, CAMERA_FAR);
		let view = DMat4::look_to_lh(DVec3::ZERO, DVec3::Y, DVec3::NEG_Z);
		let projview = proj * view;

		CameraManager {
			active_camera: Default::default(),
			default_fov,
			view,
			projview,
			sky_projview: projview,
			current_fov_y_rad: fov_y_rad,
			current_aspect_ratio: aspect_ratio,
		}
	}

	/// This function *must* be run every frame, before entities are rendered.
	pub fn update(&mut self, viewport_extent: [u32; 2], current_pos: DVec3, current_rotation: &DQuat, fov: Option<CameraFov>)
	{
		let direction = *current_rotation * DVec3::Y;
		let aspect_ratio = viewport_extent[0] as f64 / viewport_extent[1] as f64;
		let fov_y_rad = match fov.unwrap_or(self.default_fov) {
			CameraFov::X(fov_x) => fov_x / aspect_ratio,
			CameraFov::Y(fov_y) => fov_y,
		}
		.to_radians();

		let proj = DMat4::perspective_lh(fov_y_rad, aspect_ratio, CAMERA_NEAR, CAMERA_FAR);
		let view = DMat4::look_to_lh(current_pos, direction, DVec3::NEG_Z);
		let sky_view = DMat4::look_to_lh(DVec3::ZERO, direction, DVec3::NEG_Z);

		self.view = view;
		self.projview = proj * view;
		self.sky_projview = proj * sky_view;
		self.current_fov_y_rad = fov_y_rad;
		self.current_aspect_ratio = aspect_ratio;
	}

	pub fn set_active(&mut self, eid: EntityId)
	{
		self.active_camera = eid
	}
	pub fn active_camera(&self) -> EntityId
	{
		self.active_camera
	}

	/// Get the projection matrix for the current camera, but with the specified near and far planes.
	pub fn proj_with_near_far(&self, near: f64, far: f64) -> DMat4
	{
		DMat4::perspective_lh(self.current_fov_y_rad, self.current_aspect_ratio, near, far)
	}

	/// Get the current camera's view matrix.
	pub fn view(&self) -> DMat4
	{
		self.view
	}

	/// Get the multiplied projection and view matrix to be used in shaders.
	pub fn projview(&self) -> DMat4
	{
		self.projview
	}

	/// Get the multiplied projection and view matrix to be used specifically in the skybox shader.
	/// The view matrix here never changes its eye position from (0, 0, 0).
	pub fn sky_projview(&self) -> DMat4
	{
		self.sky_projview
	}
}
