/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::path::{ Path, PathBuf };
use std::sync::Arc;
use serde::Deserialize;
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use crate::render::{ RenderContext, command_buffer::CommandBuffer };
use crate::component::{ EntityComponent, DeferGpuResourceLoading, Draw };
use crate::GenericEngineError;
use crate::render::model::Model;
use crate::material::Material;

#[derive(shipyard::Component, Deserialize, EntityComponent)]
pub struct Mesh
{
	model_path: PathBuf,
	use_embedded_materials: Option<bool>,
	#[serde(skip)]
	model_data: Option<Arc<Model>>
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
impl Draw for Mesh
{
	fn draw(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		// only draw if the model has completed loading
		if let Some(model_loaded) = self.model_data.as_ref() {
			model_loaded.draw(cb)?
		}
		Ok(())
	}
}

