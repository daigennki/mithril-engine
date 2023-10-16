/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use shipyard::EntityId;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};

#[cfg(feature = "egui")]
use egui_winit_vulkano::egui;

use crate::component::EntityComponent;
use crate::material::{pbr::PBR, ColorInput, Material};
use crate::render::{model::Model, RenderContext};
use crate::GenericEngineError;

#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct Mesh
{
	pub model_path: PathBuf,
}
/*impl Mesh
{
	/// Show the egui collapsing header for this component.
	#[cfg(feature = "egui")]
	pub fn show_egui(&mut self, ui: &mut egui::Ui, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		if let Some(mat) = self.material_overrides.get_mut(0) {
			let was_none = mat.is_none();
			let mat_override = mat.get_or_insert_with(|| {
				let default_color = ColorInput::Color(Vec4::ONE);
				Box::new(PBR::new(default_color, true))
			});
			if was_none {
				mat_override.update_descriptor_set(self.model_data.as_ref().unwrap().path(), render_ctx)?;
			}

			let mut color = mat_override.get_base_color().unwrap().to_array();
			egui::CollapsingHeader::new("Mesh").show(ui, |ui| {
				ui.label("Base Color");
				ui.color_edit_button_rgba_unmultiplied(&mut color);
			});

			mat_override.set_base_color(color.into(), render_ctx)?;
		}
		Ok(())
	}
}*/

/// A single manager that manages the GPU resources for all `Mesh` components.
#[derive(shipyard::Unique)]
pub struct MeshManager
{
	resources: BTreeMap<EntityId, (Arc<Model>, Vec<Option<Box<dyn Material>>>)>
}
impl MeshManager
{
	/// Load the model for the given `Mesh`. 
	pub fn load(&mut self, render_ctx: &mut RenderContext, eid: EntityId, component: &Mesh) -> Result<(), GenericEngineError>
	{
		// model path relative to current directory
		let model_path_cd_rel = &component.model_path;
		let model_data = render_ctx.get_model(&model_path_cd_rel)?;
		let material_count = model_data.get_materials().len();

		let mut material_overrides = Vec::with_capacity(material_count);
		for _ in 0..material_count {
			material_overrides.push(None);
		}

		let existing = self.resources.insert(eid, (model_data, material_overrides));
		if existing.is_some() {
			log::warn!("`MeshManager::load` was called for an entity ID that was already loaded!");
		}

		Ok(())
	}

	/// Get a reference to the original materials of the model.
	/// This will panic if the entity ID is invalid!
	pub fn get_materials(&mut self, eid: EntityId) -> &Vec<Box<dyn Material>>
	{
		self.resources.get(&eid).unwrap().0.get_materials()
	}

	/// Get the materials that override the model's material.
	/// If a material slot is `None`, then that material slot will use the model's original material.
	/// This will panic if the entity ID is invalid!
	pub fn get_material_overrides(&mut self, eid: EntityId) -> &mut Vec<Option<Box<dyn Material>>>
	{
		&mut self.resources.get_mut(&eid).unwrap().1
	}

	/// Check if any of the materials are enabled for transparency.
	/// This will panic if the entity ID is invalid!
	pub fn has_transparency(&self, eid: EntityId) -> bool
	{
		let original_materials = self.resources.get(&eid).unwrap().0.get_materials();

		// substitute the original material if no override was specified,
		// then look for any materials with transparency enabled
		self.resources
			.get(&eid)
			.unwrap()
			.1
			.iter()
			.enumerate()
			.map(|(i, override_mat)| override_mat.as_ref().unwrap_or_else(|| &original_materials[i]))
			.any(|mat| mat.has_transparency())
	}

	/// Check if there are any materials that are *not* enabled for transparency.
	/// This will panic if loading hasn't finished yet!
	pub fn has_opaque_materials(&self, eid: EntityId) -> bool
	{
		let original_materials =  self.resources.get(&eid).unwrap().0.get_materials();

		// substitute the original material if no override was specified,
		// then look for any materials with transparency disabled
		self.resources
			.get(&eid)
			.unwrap()
			.1
			.iter()
			.enumerate()
			.map(|(i, override_mat)| override_mat.as_ref().unwrap_or_else(|| &original_materials[i]))
			.any(|mat| !mat.has_transparency())
	}

	pub fn draw(
		&self,
		eid: EntityId, 
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		projviewmodel: &Mat4,
		transparency_pass: bool
	) -> Result<(), GenericEngineError>
	{
		// only draw if the model has completed loading
		if let Some((model_loaded, material_overrides)) = self.resources.get(&eid) {
			model_loaded.draw(cb, projviewmodel, material_overrides, transparency_pass)?
		}
		Ok(())
	}
}
impl Default for MeshManager
{
	fn default() -> Self
	{
		MeshManager { resources: Default::default() }
	}
}

