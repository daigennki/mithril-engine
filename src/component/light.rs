/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use shipyard::{IntoIter, IntoWorkloadSystem, UniqueView, UniqueViewMut, View, WorkloadSystem};

use super::{camera::CameraManager, EntityComponent, Transform, WantsSystemAdded};
use crate::render::{lighting::LightManager, RenderContext};

/// These are various components that represent light sources in the world.
///

#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct DirectionalLight
{
	pub color: Vec3,
	pub intensity: f32,
}
impl WantsSystemAdded for DirectionalLight
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		None
	}
	fn add_prerender_system(&self) -> Option<WorkloadSystem>
	{
		Some(update_directional_light.into_workload_system().unwrap())
	}
}
fn update_directional_light(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut light_manager: UniqueViewMut<LightManager>,
	camera_manager: UniqueView<CameraManager>,
	dir_lights: View<DirectionalLight>,
	transforms: View<Transform>,
)
{
	// Only use the first directional light found.
	// There should really be only one of these in the world anyways.
	if let Some((dl, t)) = (&dir_lights, &transforms).iter().next() {
		// Cut the camera frustum into different pieces for the light.
		let fars = [6.0, 12.0, 24.0];
		let mut cut_frustums: [DMat4; 3] = Default::default();
		let mut near = crate::component::camera::CAMERA_NEAR;
		for (i, far) in fars.into_iter().enumerate() {
			cut_frustums[i] = camera_manager.proj_with_near_far(near, far);
			near = fars[i];
		}

		light_manager.update_dir_light(&mut render_ctx, dl, t, cut_frustums);
	}
}

/*#[derive(shipyard::Component, Deserialze, EntityComponent)]
pub struct PointLight
{
	pub color: Vec3,
	pub intensity: f32,
}

#[derive(shipyard::Component, Deserialze, EntityComponent)]
pub struct SpotLight
{
	pub color: Vec3,
	pub intensity: f32,
}*/
