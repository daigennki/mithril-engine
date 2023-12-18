/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use shipyard::{EntityId, IntoIter, IntoWithId, IntoWorkloadSystem, UniqueViewMut, View, WorkloadSystem};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use vulkano::descriptor_set::{DescriptorSet, PersistentDescriptorSet};
use vulkano::format::Format;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};

use crate::component::{EntityComponent, WantsSystemAdded};
use crate::render::{model::{Model, ModelInstance}, RenderContext};
use crate::GenericEngineError;

#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct Mesh
{
	pub model_path: PathBuf,
}
impl WantsSystemAdded for Mesh
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		None
	}
	fn add_prerender_system(&self) -> Option<WorkloadSystem>
	{
		Some(update_meshes.into_workload_system().unwrap())
	}
}
fn update_meshes(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut mesh_manager: UniqueViewMut<MeshManager>,
	meshes: View<Mesh>,
	transforms: View<super::Transform>,
)
{
	for eid in meshes.removed() {
		mesh_manager.cleanup_removed(eid);
	}

	for (eid, mesh) in meshes.inserted().iter().with_id() {
		if let Err(e) = mesh_manager.load(&mut render_ctx, eid, mesh) {
			log::error!("Failed to run `MeshManager::load`: {}", e);
		}
	}

	for (eid, (t, _)) in (transforms.inserted_or_modified(), &meshes).iter().with_id() {
		mesh_manager.set_model_matrix(eid, t.get_matrix());
	}
}

/// A single manager that manages the GPU resources for all `Mesh` components.
#[derive(Default, shipyard::Unique)]
pub struct MeshManager
{
	material_pipelines: BTreeMap<&'static str, MaterialPipelines>,

	// Loaded 3D models, with the key being the path relative to the current working directory.
	models: HashMap<PathBuf, Arc<Model>>,

	// The model and materials for each entity.
	resources: HashMap<EntityId, ModelInstance>,

	cb_3d: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,
}
impl MeshManager
{
	/// Load the model for the given `Mesh`.
	fn load(&mut self, render_ctx: &mut RenderContext, eid: EntityId, component: &Mesh) -> Result<(), GenericEngineError>
	{
		// Get a 3D model from `path`, relative to the current working directory.
		// This attempts loading if it hasn't been loaded into memory yet.
		let model_data = match self.models.get(&component.model_path) {
			Some(model) => model.clone(),
			None => {
				let new_model = Arc::new(Model::new(render_ctx, &component.model_path)?);
				self.models.insert(component.model_path.clone(), new_model.clone());
				new_model
			}
		};

		// Go through all the materials, and load the pipelines they need if they aren't already loaded.
		for mat in model_data.get_materials() {
			let mat_name = mat.material_name();
			if !self.material_pipelines.contains_key(mat_name) {
				let transparency_input_layout = render_ctx.get_transparency_renderer().get_stage3_inputs().layout().clone();
				let (opaque_pipeline, oit_pipeline) =
					mat.load_pipeline(
						render_ctx.get_material_textures_set_layout().clone(),
						render_ctx.get_light_set_layout().clone(),
						transparency_input_layout,
					)?;

				let pipeline_data = MaterialPipelines {
					opaque_pipeline,
					oit_pipeline,
				};
				self.material_pipelines.insert(mat_name, pipeline_data);
			}
		}
		
		self.resources.insert(eid, model_data.new_model_instance(render_ctx)?);

		Ok(())
	}

	fn set_model_matrix(&mut self, eid: EntityId, model_matrix: Mat4)
	{
		self.resources.get_mut(&eid).unwrap().set_model_matrix(model_matrix);
	}

	fn get_pipeline(&self, name: &str) -> Option<&Arc<GraphicsPipeline>>
	{
		self.material_pipelines.get(name).map(|pl_data| &pl_data.opaque_pipeline)
	}
	fn get_transparency_pipeline(&self, name: &str) -> Option<&Arc<GraphicsPipeline>>
	{
		self.material_pipelines
			.get(name)
			.and_then(|pl_data| pl_data.oit_pipeline.as_ref())
	}

	/// Free resources for the given entity ID. Only call this when the `Mesh` component was actually removed!
	fn cleanup_removed(&mut self, eid: EntityId)
	{
		self.resources.remove(&eid);
	}

	pub fn draw(
		&self,
		render_ctx: &RenderContext,
		projview: Mat4,
		pass_type: PassType,
		common_sets: &[Arc<PersistentDescriptorSet>],
	) -> Result<Arc<SecondaryAutoCommandBuffer>, GenericEngineError>
	{
		let (depth_format, viewport_extent, shadow_pass) = match &pass_type {
			PassType::Shadow { format, viewport_extent, .. } => (*format, *viewport_extent, true),
			_ => (crate::render::MAIN_DEPTH_FORMAT, render_ctx.swapchain_dimensions(), false),
		};

		let color_formats = pass_type.render_color_formats();
		let (pipeline, transparency_pass) = match pass_type {
			PassType::Shadow { pipeline, .. } => (pipeline, false),
			PassType::Opaque => {
				let pl = self.get_pipeline("PBR").ok_or("PBR pipeline not loaded!")?.clone();
				(pl, false)
			}
			PassType::TransparencyMoments(pl) => (pl, true),
			PassType::Transparency => {
				let pl = self
					.get_transparency_pipeline("PBR")
					.ok_or("PBR transparency pipeline not loaded!")?
					.clone();
				(pl, true)
			}
		};

		let mut cb = render_ctx.gather_commands(color_formats, Some(depth_format), None, viewport_extent)?;

		let pipeline_layout = pipeline.layout().clone();

		cb.bind_pipeline_graphics(pipeline.clone())?;

		if common_sets.len() > 0 {
			cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 1, Vec::from(common_sets))?;
		}

		for model_instance in self.resources.values() {
			model_instance.draw(&mut cb, pipeline_layout.clone(), transparency_pass, shadow_pass, &projview)?;
		}

		Ok(cb.build()?)
	}

	pub fn add_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		assert!(self.cb_3d.lock().unwrap().replace(cb).is_none())
	}

	pub fn take_cb(&mut self) -> Option<Arc<SecondaryAutoCommandBuffer>>
	{
		self.cb_3d.lock().unwrap().take()
	}
}

struct MaterialPipelines
{
	opaque_pipeline: Arc<GraphicsPipeline>,
	oit_pipeline: Option<Arc<GraphicsPipeline>>, // Optional transparency pipeline may also be specified.
}

pub enum PassType
{
	Shadow {
		pipeline: Arc<GraphicsPipeline>,
		format: Format,
		viewport_extent: [u32; 2]
	},
	Opaque,
	TransparencyMoments(Arc<GraphicsPipeline>),
	Transparency,
}
impl PassType
{
	fn render_color_formats(&self) -> &'static [Format]
	{
		match self {
			PassType::Shadow { .. } => &[],
			PassType::Opaque => &[Format::R16G16B16A16_SFLOAT],
			PassType::TransparencyMoments(_) => &[Format::R32G32B32A32_SFLOAT, Format::R32_SFLOAT, Format::R32_SFLOAT],
			PassType::Transparency => &[Format::R16G16B16A16_SFLOAT, Format::R8_UNORM],
		}
	}
}

