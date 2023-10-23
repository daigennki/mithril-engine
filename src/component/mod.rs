/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod camera;
pub mod mesh;
pub mod ui;

use glam::*;
use serde::Deserialize;
use shipyard::{EntityId, IntoIter, IntoWithId, IntoWorkloadSystem, UniqueViewMut, View, WorkloadSystem};
use std::sync::Arc;
use std::collections::BTreeMap;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::descriptor_set::{persistent::PersistentDescriptorSet, WriteDescriptorSet};

#[cfg(feature = "egui")]
use egui_winit_vulkano::egui;

use crate::render::RenderContext;
use crate::GenericEngineError;
use mithrilengine_derive::EntityComponent;

#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct Transform
{
	pub position: Vec3,
	pub scale: Vec3,
	pub rotation: Vec3,
}
impl Transform
{
	/// Calculate the quaternion for the rotation of this `Transform`.
	pub fn rotation_quat(&self) -> Quat
	{
		let rot_rad = self.rotation * std::f32::consts::PI / 180.0;
		Quat::from_euler(EulerRot::XYZ, rot_rad.x, rot_rad.y, rot_rad.z)
	}

	/// Calculate the transformation matrix for this `Transform`.
	pub fn get_matrix(&self) -> Mat4
	{
		let rot_quat = self.rotation_quat();
		Mat4::from_scale_rotation_translation(self.scale, rot_quat, self.position)
	}
}
impl WantsSystemAdded for Transform
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		Some(update_transforms.into_workload_system().unwrap())
	}
}
fn update_transforms(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut transform_manager: UniqueViewMut<TransformManager>,
	transforms: View<Transform>,
)
{
	for (eid, t) in transforms.inserted_or_modified().iter().with_id() {
		if let Err(e) = transform_manager.update(&mut render_ctx, eid, t) {
			log::error!("Failed to run `TransformManager::update`: {}", e);
		}
	}
}

/*impl Transform
{
	/// Show the egui collapsing header for this component.
	#[cfg(feature = "egui")]
	pub fn show_egui(&mut self, ui: &mut egui::Ui, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		let mut pos = self.position;
		egui::CollapsingHeader::new("Transform").show(ui, |ui| {
			if !self.is_this_static() {
				ui.columns(3, |cols| {
					cols[0].label("X");
					cols[0].add(egui::DragValue::new(&mut pos.x).speed(0.1));
					cols[1].label("Y");
					cols[1].add(egui::DragValue::new(&mut pos.y).speed(0.1));
					cols[2].label("Z");
					cols[2].add(egui::DragValue::new(&mut pos.z).speed(0.1));
				});
			}
		});
		self.set_pos(pos, render_ctx)?;

		Ok(())
	}
}*/

/// A single manager that manages the GPU resources for all `Transform` components.
#[derive(shipyard::Unique)]
pub struct TransformManager
{
	// TODO: optimize this by combining sets and buffers respectively? (reduce binds and memory fragmentation)
	sets: BTreeMap<EntityId, (Arc<PersistentDescriptorSet>, Subbuffer<Mat4>)>,
}
impl TransformManager
{
	pub fn get_descriptor_set(&self, eid: EntityId) -> Option<&Arc<PersistentDescriptorSet>>
	{
		self.sets.get(&eid).map(|(set, _)| set)
	}

	/// Update the GPU resources for the component. 
	/// This should be called whenever the component is inserted or modified.
	pub fn update(
		&mut self, 
		render_ctx: &mut RenderContext,
		eid: EntityId, 
		component: &Transform
	) -> Result<(), GenericEngineError>
	{
		let model_mat = component.get_matrix();

		if let Some((_, buf)) = self.sets.get(&eid) {
			render_ctx.copy_to_buffer(model_mat, buf.clone())?;
		} else {
			// insert a new descriptor set and buffer if they didn't exist already
			let buf = render_ctx.new_staged_buffer_from_data(model_mat, BufferUsage::UNIFORM_BUFFER)?;
			let set = render_ctx.new_descriptor_set("PBR", 0, [WriteDescriptorSet::buffer(0, buf.clone())])?;

			self.sets.insert(eid, (set, buf));
		}

		Ok(())
	}

	/// Remove the resources for the given entity ID. Invalid IDs are ignored.
	pub fn remove(&mut self, eid: EntityId)
	{
		self.sets.remove(&eid);
	}
}
impl Default for TransformManager
{
	fn default() -> Self
	{
		TransformManager { sets: Default::default() }
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
}

/// The trait that allows components to return a system relevant to themselves, which will be run every tick.
/// Every `EntityComponent` must also have this trait implemented, even if it doesn't need to add any systems.
///
/// NOTE: The caveat with this is that the system will only be added if the component is specified in the map file!
/// The system won't be added if you insert the component through the program. (TODO: figure out a way to get the
/// system added even when the component is added through the program)
pub trait WantsSystemAdded
{
	fn add_system(&self) -> Option<WorkloadSystem>;
}

