/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use shipyard::{EntityId, WorkloadSystem};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::PipelineLayout;

#[cfg(feature = "egui")]
use egui_winit_vulkano::egui;

use crate::component::{EntityComponent, WantsSystemAdded};
use crate::material::{/*pbr::PBR, ColorInput,*/ Material};
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
impl WantsSystemAdded for Mesh
{
	fn add_system(&self) -> Option<(std::any::TypeId, WorkloadSystem)>
	{
		None
	}
}

/// A single manager that manages the GPU resources for all `Mesh` components.
#[derive(shipyard::Unique)]
pub struct MeshManager
{
	// Loaded 3D models, with the key being the path relative to the current working directory.
	models: BTreeMap<PathBuf, Arc<Model>>,

	// The models and material overrides for each entity.
	resources: BTreeMap<EntityId, (Arc<Model>, Vec<(Option<Box<dyn Material>>, Arc<PersistentDescriptorSet>)>)>
}
impl MeshManager
{
	/// Load the model for the given `Mesh`. 
	pub fn load(&mut self, render_ctx: &mut RenderContext, eid: EntityId, component: &Mesh) -> Result<(), GenericEngineError>
	{
		// Get a 3D model from `path`, relative to the current working directory.
		// This attempts loading if it hasn't been loaded into memory yet.
		let model_data = match self.models.get(&component.model_path) {
			Some(model) => {
				log::info!("Reusing loaded model '{}'", component.model_path.display());
				model.clone()
			}
			None => {
				let new_model = Arc::new(Model::new(render_ctx, &component.model_path)?);
				self.models.insert(component.model_path.clone(), new_model.clone());
				new_model
			}
		};

		let orig_materials = model_data.get_materials();
		let material_count = orig_materials.len();

		let mut material_overrides = Vec::with_capacity(material_count);
		for mat in orig_materials {
			// TODO: if there's a material override, use that instead of the original material
			// to create the descriptor set
			let writes = mat.gen_descriptor_set_writes(&component.model_path, render_ctx)?;
			let set = render_ctx.new_descriptor_set(mat.pipeline_name(), 1, writes)?;

			material_overrides.push((None, set));
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
	pub fn get_material_overrides(&mut self, eid: EntityId) -> &mut Vec<(Option<Box<dyn Material>>, Arc<PersistentDescriptorSet>)>
	{
		&mut self.resources.get_mut(&eid).unwrap().1
	}

	/// Check if any of the materials are enabled for transparency.
	/// This may also return false if the entity ID is invalid.
	pub fn has_transparency(&self, eid: EntityId) -> bool
	{
		// substitute the original material if no override was specified,
		// then look for any materials with transparency enabled
		match self.resources.get(&eid) {
			Some((model, material_overrides)) => {
				let original_materials = model.get_materials();
				material_overrides
					.iter()
					.enumerate()
					.map(|(i, (override_mat, _))| override_mat.as_ref().unwrap_or_else(|| &original_materials[i]))
					.any(|mat| mat.has_transparency())
			}
			None => false
		}
	}

	/// Check if there are any materials that are *not* enabled for transparency.
	/// This may also return false if the entity ID is invalid.
	pub fn has_opaque_materials(&self, eid: EntityId) -> bool
	{
		// substitute the original material if no override was specified,
		// then look for any materials with transparency disabled
		match self.resources.get(&eid) {
			Some((model, material_overrides)) => {
				let original_materials = model.get_materials();
				material_overrides
					.iter()
					.enumerate()
					.map(|(i, (override_mat, _))| override_mat.as_ref().unwrap_or_else(|| &original_materials[i]))
					.any(|mat| !mat.has_transparency())
			}
			None => false
		}
	}

	pub fn draw(
		&self,
		eid: EntityId, 
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		pipeline_layout: Arc<PipelineLayout>,
		projviewmodel: &Mat4,
		transparency_pass: bool
	) -> Result<(), GenericEngineError>
	{
		// only draw if the model has completed loading
		if let Some((model_loaded, material_overrides_and_sets)) = self.resources.get(&eid) {
			model_loaded.draw(cb, pipeline_layout, projviewmodel, material_overrides_and_sets, transparency_pass)?
		}
		Ok(())
	}
}
impl Default for MeshManager
{
	fn default() -> Self
	{
		MeshManager {
			models: Default::default(),
			resources: Default::default(),
		}
	}
}

