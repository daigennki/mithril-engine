/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod camera;
pub mod light;
pub mod mesh;
pub mod physics;
pub mod ui;

use glam::*;
use serde::Deserialize;
use shipyard::{IntoIter, IntoWorkloadSystem, ViewMut, WorkloadSystem};

use mithrilengine_derive::EntityComponent;
use crate::SystemBundle;

/// A component representing transformation characteristics of an entity.
///
/// NOTE: Rotation gets wrapped if it goes outside the [-360.0, 360.0] degress range.
#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct Transform
{
	pub position: DVec3,
	pub scale: DVec3,
	pub rotation: DVec3,
}
impl Transform
{
	/// Calculate the quaternion for the rotation of this `Transform`.
	pub fn rotation_quat(&self) -> DQuat
	{
		let rot_rad = self.rotation * std::f64::consts::PI / 180.0;
		DQuat::from_euler(EulerRot::ZXY, rot_rad.z, rot_rad.x, rot_rad.y)
	}

	/// Calculate the affine transformation for this `Transform`.
	pub fn get_affine(&self) -> DAffine3
	{
		let quat = self.rotation_quat();
		DAffine3::from_scale_rotation_translation(self.scale, quat, self.position)
	}
}
impl WantsSystemAdded for Transform
{
	fn add_system() -> Option<WorkloadSystem>
	{
		Some(wrap_rotation.into_workload_system().unwrap())
	}
	fn add_prerender_system() -> Option<WorkloadSystem>
	{
		None
	}
}

// Wrap rotation values to make them stay within range of [-360.0, 360.0] exclusive.
fn wrap_rotation(mut transforms: ViewMut<Transform>)
{
	for mut t in transforms.inserted_or_modified_mut().iter() {
		if t.rotation.abs().max_element() >= 360.0 {
			t.rotation %= 360.0;
		}
	}
}

/// The trait that every component to be used in a map file must implement.
/// This allows a deserialized component to add itself to the world, so that it can be
/// deserialized as `Box<dyn EntityComponent>` but still keep its concrete type in the world.
#[typetag::deserialize]
pub trait EntityComponent: WantsSystemAdded + Send + Sync
{
	fn add_to_entity(self: Box<Self>, world: &mut shipyard::World, eid: shipyard::EntityId);

	fn type_id(&self) -> std::any::TypeId;

	fn type_name(&self) -> &'static str;
}

/// The trait that allows components to return a system relevant to themselves, which will be run
/// every tick. Every `EntityComponent` must also have this trait implemented, even if it doesn't
/// need to add any systems.
///
/// NOTE: This might have some issues with changes made to components from other components not
/// becoming visible until the next frame. (TODO: somehow let some components to update later than
/// others)
pub trait WantsSystemAdded
{
	fn add_system() -> Option<WorkloadSystem>
	where
		Self: Sized;

	fn add_prerender_system() -> Option<WorkloadSystem>
	where
		Self: Sized;
}
