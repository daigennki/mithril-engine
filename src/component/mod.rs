/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod camera;
pub mod light;
pub mod mesh;
pub mod ui;

use glam::*;
use serde::Deserialize;
use shipyard::{/*IntoIter, IntoWorkloadSystem, ViewMut,*/ WorkloadSystem};

use mithrilengine_derive::EntityComponent;

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

	/// Calculate the transformation matrix for this `Transform`.
	pub fn get_matrix(&self) -> DMat4
	{
		let rot_quat = self.rotation_quat();
		DMat4::from_scale_rotation_translation(self.scale, rot_quat, self.position)
	}
}
impl WantsSystemAdded for Transform
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		//Some(wrap_rotation.into_workload_system().unwrap())
		None
	}
	fn add_prerender_system(&self) -> Option<WorkloadSystem>
	{
		None
	}
}

// Wrap rotation values to make them stay within range of [-180.0, 180.0] inclusive.
/*fn wrap_rotation(mut transforms: ViewMut<Transform>)
{
	for mut t in (&mut transforms).inserted_or_modified() {
		if t.rotation.max_element() > 180.0 {
			t.rotation = (t.rotation % 180.0) - 180.0;
		} else if t.rotation.min_element() < -180.0 {
			t.rotation = (t.rotation % 180.0) + 180.0;
		}
	}
}*/

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

/// The trait that allows components to return a system relevant to themselves, which will be run every tick.
/// Every `EntityComponent` must also have this trait implemented, even if it doesn't need to add any systems.
///
/// NOTE: The caveat with this is that the system will only be added if the component is specified in the map file!
/// The system won't be added if you insert the component through the program. (TODO: figure out a way to get the
/// system added even when the component is added through the program)
///
/// NOTE: This might also have some issues with changes made to components from other components not becoming visible
/// until the next frame. (TODO: add a " update" system for components that need to make changes earlier, or instead
/// make some components update later than others)
pub trait WantsSystemAdded
{
	fn add_system(&self) -> Option<WorkloadSystem>;

	fn add_prerender_system(&self) -> Option<WorkloadSystem>;
}
