/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};

#[cfg(feature = "egui")]
use egui_winit_vulkano::egui;

use crate::component::{DeferGpuResourceLoading, EntityComponent};
use crate::material::{pbr::PBR, ColorInput, Material};
use crate::render::{model::Model, RenderContext};
use crate::GenericEngineError;

#[derive(shipyard::Component, Deserialize, EntityComponent)]
pub struct Mesh
{
	model_path: PathBuf,

	#[serde(skip)]
	model_data: Option<Arc<Model>>,
	#[serde(skip)]
	material_overrides: Vec<Option<Box<dyn Material>>>,
}
impl Mesh
{
	/// Get a reference to the original materials of the model.
	pub fn get_materials(&mut self) -> Option<&Vec<Box<dyn Material>>>
	{
		Arc::<Model>::get_mut(self.model_data.as_mut().unwrap()).map(|m| m.get_materials())
	}

	/// Get the materials that override the model's material.
	/// If a material slot is `None`, then that material slot will use the model's original material.
	pub fn get_material_overrides(&mut self) -> &mut Vec<Option<Box<dyn Material>>>
	{
		&mut self.material_overrides
	}

	/// Check if any of the materials are enabled for transparency.
	/// This will panic if loading hasn't finished yet!
	pub fn has_transparency(&self) -> bool
	{
		let original_materials = self.model_data.as_ref().unwrap().get_materials();

		// substitute the original material if no override was specified,
		// then look for any materials with transparency enabled
		self.material_overrides
			.iter()
			.enumerate()
			.map(|(i, override_mat)| override_mat.as_ref().unwrap_or_else(|| &original_materials[i]))
			.any(|mat| mat.has_transparency())
	}

	/// Check if there are any materials that are *not* enabled for transparency.
	/// This will panic if loading hasn't finished yet!
	pub fn has_opaque_materials(&self) -> bool
	{
		let original_materials = self.model_data.as_ref().unwrap().get_materials();

		// substitute the original material if no override was specified,
		// then look for any materials with transparency disabled
		self.material_overrides
			.iter()
			.enumerate()
			.map(|(i, override_mat)| override_mat.as_ref().unwrap_or_else(|| &original_materials[i]))
			.any(|mat| !mat.has_transparency())
	}

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		projviewmodel: &Mat4,
		transparency_pass: bool,
	) -> Result<(), GenericEngineError>
	{
		// only draw if the model has completed loading
		if let Some(model_loaded) = self.model_data.as_ref() {
			model_loaded.draw(cb, projviewmodel, &self.material_overrides, transparency_pass)?
		}
		Ok(())
	}

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
}
impl DeferGpuResourceLoading for Mesh
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		// model path relative to current directory
		let model_path_cd_rel = &self.model_path;
		let model_data = render_ctx.get_model(&model_path_cd_rel)?;
		let material_count = model_data.get_materials().len();

		self.model_data = Some(model_data);

		self.material_overrides = Vec::with_capacity(material_count);
		for _ in 0..material_count {
			self.material_overrides.push(None);
		}

		Ok(())
	}
}
