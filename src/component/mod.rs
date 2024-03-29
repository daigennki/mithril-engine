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
use shipyard::{EntityId, IntoIter, IntoWorkloadSystem, ViewMut, WorkloadSystem, World};

use crate::SystemBundle;
use mithril_engine_derive::EntityComponent;

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
impl ComponentSystems for Transform
{
	fn late_update() -> Option<WorkloadSystem>
	{
		Some(wrap_rotation.into_workload_system().unwrap())
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

/// The trait that every component to be used in a map must implement. Implement this using
/// `#[derive(EntityComponent)]`.
#[typetag::deserialize]
pub trait EntityComponent: ComponentSystems + Send + Sync
{
	/// Allows a deserialized component to add itself to the world, so that it can be deserialized
	/// as `Box<dyn EntityComponent>` but still keep its concrete type in the world.
	fn add_to_entity(self: Box<Self>, world: &mut World, eid: EntityId);
}

/// The trait that allows components to return systems relevant to themselves. Every
/// `EntityComponent` must have this trait implemented, even if it doesn't need to add any systems.
pub trait ComponentSystems
{
	/// Run before physics updates every tick.
	fn update() -> Option<WorkloadSystem>
	where
		Self: Sized,
	{
		None
	}

	/// Run after physics updates every tick.
	fn late_update() -> Option<WorkloadSystem>
	where
		Self: Sized,
	{
		None
	}
}
