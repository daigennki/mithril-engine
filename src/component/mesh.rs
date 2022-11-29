/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use egui_winit_vulkano::egui;
use glam::*;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};

use crate::component::{DeferGpuResourceLoading, EntityComponent};
use crate::material::{ColorInput, Material, pbr::PBR};
use crate::render::{model::Model, RenderContext};
use crate::GenericEngineError;

#[derive(shipyard::Component, Deserialize, EntityComponent)]
pub struct Mesh
{
	model_path: PathBuf,
	#[serde(default)]
	use_embedded_materials: bool,
	#[serde(default)]
	transparent: bool,

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

	pub fn using_embedded_materials(&self) -> bool
	{
		self.use_embedded_materials
	}

	pub fn has_transparency(&self) -> bool
	{
		self.transparent
	}

	pub fn draw(
		&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, projviewmodel: &Mat4,
	) -> Result<(), GenericEngineError>
	{
		// only draw if the model has completed loading
		if let Some(model_loaded) = self.model_data.as_ref() {
			model_loaded.draw(cb, projviewmodel, &self.material_overrides)?
		}
		Ok(())
	}

	/// Show the egui collapsing header for this component.
	pub fn show_egui(&mut self, ui: &mut egui::Ui, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		if let Some(mat) = self.material_overrides.get_mut(0) {
			let was_none = mat.is_none();
			let mat_override = mat.get_or_insert_with(|| {
				let default_color = ColorInput::Color(Vec4::ONE);
				Box::new(PBR::new(default_color))
			});
			if was_none {
				mat_override.update_descriptor_set(self.model_data.as_ref().unwrap().path(), render_ctx)?;
			}

			let mut color = mat_override
				.get_base_color()
				.unwrap()
				.to_array();
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
		let model_path_cd_rel = Path::new("./models/").join(&self.model_path);
		let model_data = render_ctx.get_model(&model_path_cd_rel, self.use_embedded_materials)?;
		let material_count = model_data.get_materials().len();

		self.model_data = Some(model_data);
		
		self.material_overrides = Vec::with_capacity(material_count);
		for _ in 0..material_count {
			self.material_overrides.push(None);
		}

		Ok(())
	}
}
