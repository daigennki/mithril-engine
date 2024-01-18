/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use rapier3d_f64::prelude::{ColliderBuilder, RigidBodyType};
use serde::Deserialize;
use shipyard::{EntityId, Get, IntoIter, IntoWithId, IntoWorkloadSystem, UniqueViewMut, View, ViewMut, WorkloadSystem};
use std::collections::HashMap;

use crate::component::{EntityComponent, WantsSystemAdded};

const DEFAULT_GRAVITY: DVec3 = DVec3 {
	x: 0.0,
	y: 0.0,
	z: -9.81,
};

#[derive(Deserialize, Copy, Clone, Debug)]
pub enum ColliderType
{
	Cuboid
	{
		half_extents: DVec3
	},
	Ball
	{
		radius: f64
	},
}
impl ColliderType
{
	fn collider_builder(&self) -> rapier3d_f64::prelude::ColliderBuilder
	{
		match self {
			Self::Cuboid { half_extents } => ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z),
			Self::Ball { radius } => ColliderBuilder::ball(*radius),
		}
	}
}

#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct Collider
{
	pub collider_type: ColliderType,
	pub mass: f64,

	#[serde(default)]
	pub restitution: f64,
}
impl WantsSystemAdded for Collider
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		None
	}
	fn add_prerender_system(&self) -> Option<WorkloadSystem>
	{
		None
	}
}

#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct RigidBody
{
	pub rigid_body_type: RigidBodyType,
}
impl WantsSystemAdded for RigidBody
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		Some(simulate_physics.into_workload_system().unwrap())
	}
	fn add_prerender_system(&self) -> Option<WorkloadSystem>
	{
		None
	}
}
fn simulate_physics(
	mut physics_manager: UniqueViewMut<PhysicsManager>,
	mut transforms: ViewMut<super::Transform>,
	rigid_body_components: View<RigidBody>,
	collider_components: View<Collider>,
)
{
	// Insert rigid bodies for newly inserted `RigidBody` components
	for (eid, (t, rb)) in (&transforms, rigid_body_components.inserted()).iter().with_id() {
		let position = nalgebra::geometry::Isometry {
			rotation: t.rotation_quat().into(),
			translation: t.position.into(),
		};
		let rigid_body = rapier3d_f64::prelude::RigidBodyBuilder::new(rb.rigid_body_type)
			.position(position)
			.user_data(eid.inner() as u128)
			.build();
		let rigid_body_handle = physics_manager.rigid_body_set.insert(rigid_body);
		physics_manager.rigid_body_handles.insert(eid, rigid_body_handle);
	}

	// Insert colliders for newly inserted `Collider` components
	for (eid, c) in collider_components.inserted().iter().with_id() {
		physics_manager.insert_collider(eid, c.collider_type, c.mass, c.restitution);
	}

	// TODO: handle changes by other components to the `Transform` used by a `RigidBody` component

	// TODO: clean up removed components

	physics_manager.step();

	// Reflect changes made by the physics engine to the `Transform` component
	for (_, rb) in physics_manager.rigid_body_set.iter() {
		if !rb.is_sleeping() && rb.body_type() != RigidBodyType::Fixed {
			let eid = EntityId::from_inner(rb.user_data as u64).unwrap();
			if let Ok(mut t) = (&mut transforms).get(eid) {
				let (pos, quat) = (*rb.position()).into();
				let rotation_rad: DVec3 = quat.to_euler(EulerRot::ZXY).into();
				t.position = pos;
				t.rotation = rotation_rad * 180.0 / std::f64::consts::PI;
			}
		}
	}
}

#[derive(shipyard::Unique, Default)]
pub struct PhysicsManager
{
	rigid_body_set: rapier3d_f64::prelude::RigidBodySet,
	rigid_body_handles: HashMap<EntityId, rapier3d_f64::prelude::RigidBodyHandle>,
	collider_set: rapier3d_f64::prelude::ColliderSet,
	collider_handles: HashMap<EntityId, rapier3d_f64::prelude::ColliderHandle>,

	integration_parameters: rapier3d_f64::prelude::IntegrationParameters,
	physics_pipeline: rapier3d_f64::prelude::PhysicsPipeline,
	island_manager: rapier3d_f64::prelude::IslandManager,
	broad_phase: rapier3d_f64::prelude::BroadPhase,
	narrow_phase: rapier3d_f64::prelude::NarrowPhase,
	impulse_joint_set: rapier3d_f64::prelude::ImpulseJointSet,
	multibody_joint_set: rapier3d_f64::prelude::MultibodyJointSet,
	ccd_solver: rapier3d_f64::prelude::CCDSolver,
}
impl PhysicsManager
{
	fn insert_collider(&mut self, eid: EntityId, collider_type: ColliderType, mass: f64, restitution: f64)
	{
		if let Some(rigid_body_handle) = self.rigid_body_handles.get(&eid).copied() {
			let collider = collider_type
				.collider_builder()
				.mass(mass)
				.restitution(restitution)
				.user_data(eid.inner() as u128)
				.build();
			let collider_handle = self
				.collider_set
				.insert_with_parent(collider, rigid_body_handle, &mut self.rigid_body_set);
			self.collider_handles.insert(eid, collider_handle);
		}
	}

	fn step(&mut self)
	{
		self.physics_pipeline.step(
			&DEFAULT_GRAVITY.into(),
			&self.integration_parameters,
			&mut self.island_manager,
			&mut self.broad_phase,
			&mut self.narrow_phase,
			&mut self.rigid_body_set,
			&mut self.collider_set,
			&mut self.impulse_joint_set,
			&mut self.multibody_joint_set,
			&mut self.ccd_solver,
			None,
			&(),
			&(),
		);
	}
}
