/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use shipyard::{EntityId, IntoIter, IntoWithId, IntoWorkloadSystem, UniqueViewMut, View, WorkloadSystem};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::{layout::DescriptorSetLayout, PersistentDescriptorSet};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout};

use crate::component::{EntityComponent, WantsSystemAdded};
use crate::material::Material;
use crate::render::{model::Model, RenderContext};
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
	for (eid, mesh) in meshes.inserted().iter().with_id() {
		if let Err(e) = mesh_manager.load(&mut render_ctx, eid, mesh) {
			log::error!("Failed to run `MeshManager::load`: {}", e);
		}
	}

	for (eid, (t, _)) in (transforms.inserted_or_modified(), &meshes).iter().with_id() {
		mesh_manager.set_model_matrix(eid, t.get_matrix());
	}

	for eid in meshes.removed() {
		mesh_manager.cleanup_removed(eid);
	}
}

pub struct MaterialResources
{
	pub mat_override: Option<Box<dyn Material>>,
	pub mat_set: Arc<PersistentDescriptorSet>,
	pub mat_basecolor_only_set: Arc<PersistentDescriptorSet>, // used for OIT when only "base color" texture is needed
}

struct MeshResources
{
	model: Arc<Model>,
	material_resources: Vec<MaterialResources>,
	model_matrix: Mat4,
}

/// Pipeline for each material, as well as the descriptor set layout unique to each material.
struct PipelineData
{
	opaque_pipeline: Arc<GraphicsPipeline>,
	oit_pipeline: Option<Arc<GraphicsPipeline>>, // Optional transparency pipeline may also be specified.
	material_set_layout: Arc<DescriptorSetLayout>,
}

#[derive(Clone, Copy, bytemuck::AnyBitPattern)]
#[repr(C)]
pub struct MeshPushConstant
{
	projviewmodel: Mat4,
	model_x: Vec4,
	model_y: Vec4,
	model_z: Vec4,
}

/// A single manager that manages the GPU resources for all `Mesh` components.
#[derive(shipyard::Unique)]
pub struct MeshManager
{
	material_pipelines: HashMap<&'static str, PipelineData>,

	// Loaded 3D models, with the key being the path relative to the current working directory.
	models: BTreeMap<PathBuf, Arc<Model>>,

	basecolor_only_set_layout: Arc<DescriptorSetLayout>, // Set layout for OIT when only "base color" texture is needed
	transparency_input_layout: Arc<DescriptorSetLayout>,
	light_set_layout: Arc<DescriptorSetLayout>,

	// The models and materials for each entity.
	resources: HashMap<EntityId, MeshResources>,

	cb_3d: Mutex<VecDeque<Arc<SecondaryAutoCommandBuffer>>>,
}
impl MeshManager
{
	pub fn new(
		basecolor_only_set_layout: Arc<DescriptorSetLayout>,
		transparency_input_layout: Arc<DescriptorSetLayout>,
		light_set_layout: Arc<DescriptorSetLayout>,
	) -> Self
	{
		MeshManager {
			material_pipelines: Default::default(),
			models: Default::default(),
			basecolor_only_set_layout,
			transparency_input_layout,
			light_set_layout,
			resources: Default::default(),
			cb_3d: Mutex::new(VecDeque::with_capacity(2)),
		}
	}

	/// Load the model for the given `Mesh`.
	pub fn load(&mut self, render_ctx: &mut RenderContext, eid: EntityId, component: &Mesh) -> Result<(), GenericEngineError>
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

		let orig_materials = model_data.get_materials();
		let material_count = orig_materials.len();

		let mut material_resources = Vec::with_capacity(material_count);
		for mat in orig_materials {
			// TODO: if there's a material override, use that instead of the original material
			// to create the descriptor set

			let parent_folder = component.model_path.parent().unwrap();

			let mat_name = mat.material_name();

			// Load the pipeline if it hasn't already been loaded.
			if !self.material_pipelines.contains_key(mat_name) {
				let (opaque_pipeline, oit_pipeline, material_set_layout) =
					mat.load_pipeline(self.light_set_layout.clone(), self.transparency_input_layout.clone())?;

				let pipeline_data = PipelineData {
					opaque_pipeline,
					oit_pipeline,
					material_set_layout,
				};
				self.material_pipelines.insert(mat_name, pipeline_data);
			}

			// We use `unwrap` here since the material pipeline must've been loaded above.
			let set_layout = self.material_pipelines.get(mat_name).unwrap().material_set_layout.clone();

			let writes = mat.gen_descriptor_set_writes(parent_folder, render_ctx)?;
			let mat_set = PersistentDescriptorSet::new(render_ctx.descriptor_set_allocator(), set_layout, writes, [])?;

			let base_color_writes = mat.gen_base_color_descriptor_set_writes(parent_folder, render_ctx)?;
			let mat_basecolor_only_set = PersistentDescriptorSet::new(
				render_ctx.descriptor_set_allocator(),
				self.basecolor_only_set_layout.clone(),
				base_color_writes,
				[],
			)?;

			material_resources.push(MaterialResources {
				mat_override: None,
				mat_set,
				mat_basecolor_only_set,
			});
		}

		let mesh_resources = MeshResources {
			model: model_data,
			material_resources,
			model_matrix: Default::default(),
		};
		let existing = self.resources.insert(eid, mesh_resources);
		if existing.is_some() {
			log::warn!("`MeshManager::load` was called for an entity ID that was already loaded!");
		}

		Ok(())
	}

	pub fn set_model_matrix(&mut self, eid: EntityId, model_matrix: Mat4)
	{
		self.resources.get_mut(&eid).unwrap().model_matrix = model_matrix;
	}

	pub fn get_pipeline(&self, name: &str) -> Option<&Arc<GraphicsPipeline>>
	{
		self.material_pipelines.get(name).map(|pl_data| &pl_data.opaque_pipeline)
	}
	pub fn get_transparency_pipeline(&self, name: &str) -> Option<&Arc<GraphicsPipeline>>
	{
		self.material_pipelines
			.get(name)
			.and_then(|pl_data| pl_data.oit_pipeline.as_ref())
	}

	/// Get a reference to the original materials of the model.
	/// This will panic if the entity ID is invalid!
	pub fn get_materials(&mut self, eid: EntityId) -> &Vec<Box<dyn Material>>
	{
		self.resources.get(&eid).unwrap().model.get_materials()
	}

	/*/// Get the materials that override the model's material.
	/// If a material slot is `None`, then that material slot will use the model's original material.
	/// This will panic if the entity ID is invalid!
	pub fn get_material_overrides(&mut self, eid: EntityId) -> &mut Vec<(Option<Box<dyn Material>>, Arc<PersistentDescriptorSet>)>
	{
		&mut self.resources.get_mut(&eid).unwrap().1
	}*/

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		projview: Mat4,
		transparency_pass: bool,
		base_color_pipeline_layout: Option<Arc<PipelineLayout>>,
		shadow_pass_pipeline_layout: Option<Arc<PipelineLayout>>,
		common_sets: &[Arc<PersistentDescriptorSet>],
	) -> Result<(), GenericEngineError>
	{
		let pipeline_layout;
		if let Some(shadow_pl_layout) = shadow_pass_pipeline_layout.clone() {
			pipeline_layout = shadow_pl_layout;
		} else if let Some(some_base_color_pl_layout) = base_color_pipeline_layout.clone() {
			pipeline_layout = some_base_color_pl_layout;
		} else {
			let pipeline = if transparency_pass {
				self.get_transparency_pipeline("PBR")
					.ok_or("PBR transparency pipeline not loaded!")?
			} else {
				self.get_pipeline("PBR").ok_or("PBR pipeline not loaded!")?
			};
			pipeline_layout = pipeline.layout().clone();

			cb.bind_pipeline_graphics(pipeline.clone())?;
			cb.bind_descriptor_sets(
				PipelineBindPoint::Graphics,
				pipeline_layout.clone(),
				1,
				Vec::from(common_sets),
			)?;
		}

		// go through all the loaded meshes
		for mesh_resources in self.resources.values() {
			let original_materials = mesh_resources.model.get_materials();

			// substitute the original material if no override was specified,
			// then look for any materials with transparency enabled or disabled (depending on `transparency_pass`)
			let continue_draw = mesh_resources
				.material_resources
				.iter()
				.enumerate()
				.map(|(i, res)| res.mat_override.as_ref().unwrap_or_else(|| &original_materials[i]))
				.any(|mat| mat.has_transparency() == transparency_pass);

			if continue_draw {
				let shadow_pass = shadow_pass_pipeline_layout.is_some();
				let base_color_only = base_color_pipeline_layout.is_some();

				let projviewmodel = projview * mesh_resources.model_matrix;
				if shadow_pass {
					// TODO: also consider point lights, which require different matrices
					cb.push_constants(pipeline_layout.clone(), 0, projviewmodel)?;
				} else {
					let model_matrix = mesh_resources.model_matrix;
					let translation = model_matrix.w_axis.xyz();
					let push_data = MeshPushConstant {
						projviewmodel,
						model_x: model_matrix.x_axis.xyz().extend(translation.x),
						model_y: model_matrix.y_axis.xyz().extend(translation.y),
						model_z: model_matrix.z_axis.xyz().extend(translation.z),
					};
					cb.push_constants(pipeline_layout.clone(), 0, push_data)?;
				}

				mesh_resources.model.draw(
					cb,
					pipeline_layout.clone(),
					&projviewmodel,
					&mesh_resources.material_resources,
					transparency_pass,
					base_color_only,
					shadow_pass,
				)?;
			}
		}
		Ok(())
	}

	/// Free resources for the given entity ID. Only call this when the `Mesh` component was actually removed!
	pub fn cleanup_removed(&mut self, eid: EntityId)
	{
		self.resources.remove(&eid);
	}

	pub fn add_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		self.cb_3d.lock().unwrap().push_back(cb);
	}

	pub fn take_cb(&mut self) -> VecDeque<Arc<SecondaryAutoCommandBuffer>>
	{
		std::mem::replace(&mut self.cb_3d.lock().unwrap(), VecDeque::with_capacity(2))
	}
}
