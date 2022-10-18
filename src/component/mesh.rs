/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use egui_winit_vulkano::egui;
use glam::*;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::command_buffer::SecondaryAutoCommandBuffer;

use crate::component::{DeferGpuResourceLoading, EntityComponent};
use crate::material::Material;
use crate::render::model::Model;
use crate::render::{command_buffer::CommandBuffer, RenderContext};
use crate::GenericEngineError;


#[derive(shipyard::Component, Deserialize, EntityComponent)]
pub struct Mesh
{
	model_path: PathBuf,
	use_embedded_materials: Option<bool>,
	#[serde(skip)]
	model_data: Option<Arc<Model>>,
}
impl Mesh
{
	pub fn get_materials(&mut self) -> Option<&mut Vec<Box<dyn Material>>>
	{
		Arc::<Model>::get_mut(self.model_data.as_mut().unwrap()).map(|m| m.get_materials())
	}

	pub fn using_embedded_materials(&self) -> bool
	{
		self.use_embedded_materials.unwrap_or(false)
	}

	pub fn draw(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>, projviewmodel: &Mat4) -> Result<(), GenericEngineError>
	{
		// only draw if the model has completed loading
		if let Some(model_loaded) = self.model_data.as_ref() {
			model_loaded.draw(cb, projviewmodel)?
		}
		Ok(())
	}

	/// Show the egui collapsing header for this component.
	pub fn show_egui(&mut self, ui: &mut egui::Ui, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		if let Some(materials) = self.get_materials() {
			let mat = &mut materials[0];
			let mut color = mat.get_base_color().to_array();
			egui::CollapsingHeader::new("Mesh")
				.show(ui, |ui| {
					ui.label("Base Color");
					ui.color_edit_button_rgba_unmultiplied(&mut color);
				});
			mat.set_base_color(color.into(), render_ctx)?;
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
		self.model_data = Some(render_ctx.get_model(&model_path_cd_rel, self.use_embedded_materials.unwrap_or(false))?);
		Ok(())
	}
}

