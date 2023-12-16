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
use vulkano::descriptor_set::{
	layout::DescriptorSetLayout, PersistentDescriptorSet, WriteDescriptorSet, WriteDescriptorSetElements
};
use vulkano::format::Format;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};

use crate::component::{EntityComponent, WantsSystemAdded};
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
	pub mat_set: Arc<PersistentDescriptorSet>,
	pub mat_basecolor_only_set: Arc<PersistentDescriptorSet>, // used for OIT when only "base color" texture is needed
}

struct MeshResources
{
	model: Arc<Model>,
	material_resources: BTreeMap<&'static str, MaterialResources>,
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

	cb_3d: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,
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
			cb_3d: Default::default(),
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

		// Get the sorted writes for each material, grouped by their respective shaders.
		let mut material_write_groups = BTreeMap::new();
		for (_, mat) in orig_materials {
			let mat_name = mat.material_name();
			let parent_folder = component.model_path.parent().unwrap();

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

			let write = mat.gen_descriptor_set_write(parent_folder, render_ctx)?;
			let base_color_write = mat.gen_base_color_descriptor_set_write(parent_folder, render_ctx)?;

			if !material_write_groups.contains_key(mat_name) {
				material_write_groups.insert(mat_name, (Vec::new(), Vec::new(), set_layout));
			}
	
			let (existing_writes, existing_base_color_writes, _) = material_write_groups.get_mut(mat_name).unwrap();
			existing_writes.push(write);
			existing_base_color_writes.push(base_color_write);
		}

		let mut material_resources = BTreeMap::new();
		for (mat_name, (writes, base_color_writes, set_layout)) in material_write_groups {
			// Reduce the writes into a single array write, and create the descriptor sets.
			let merged_write = merge_descriptor_writes(writes);
			let merged_base_color_write = merge_descriptor_writes(base_color_writes);

			// TODO: also get a separate count for the "base color-only" set,
			// to account for shaders that take multiple textures for each material
			let variable_descriptor_count = match merged_write.elements() {
				WriteDescriptorSetElements::ImageView(contents) => contents.len().try_into()?,
				_ => unimplemented!(),
			};
			log::debug!("variable descriptor count: {}", variable_descriptor_count);

			let mat_set = PersistentDescriptorSet::new_variable(
				render_ctx.descriptor_set_allocator(),
				set_layout,
				variable_descriptor_count,
				[merged_write],
				[]
			)?;
			let mat_basecolor_only_set = PersistentDescriptorSet::new_variable(
				render_ctx.descriptor_set_allocator(),
				self.basecolor_only_set_layout.clone(),
				variable_descriptor_count,
				[merged_base_color_write],
				[],
			)?;

			material_resources.insert(
				mat_name,
				MaterialResources {
					mat_set,
					mat_basecolor_only_set,
				},
			);
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

	/*/// Get a reference to the sorted materials of the model.
	/// This will panic if the entity ID is invalid!
	pub fn get_materials(&mut self, eid: EntityId) -> &Vec<(usize, Box<dyn Material>)>
	{
		self.resources.get(&eid).unwrap().model.get_materials()
	}*/

	/*/// Get the materials that override the model's material.
	/// If a material slot is `None`, then that material slot will use the model's original material.
	/// This will panic if the entity ID is invalid!
	pub fn get_material_overrides(&mut self, eid: EntityId) -> &mut Vec<(Option<Box<dyn Material>>, Arc<PersistentDescriptorSet>)>
	{
		&mut self.resources.get_mut(&eid).unwrap().1
	}*/

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
		let (pipeline, transparency_pass, base_color_only) = match pass_type {
			PassType::Shadow { pipeline, .. } => (pipeline, false, false),
			PassType::Opaque => {
				let pl = self.get_pipeline("PBR").ok_or("PBR pipeline not loaded!")?.clone();
				(pl, false, false)
			}
			PassType::TransparencyMoments(pl) => (pl, true, true),
			PassType::Transparency => {
				let pl = self
					.get_transparency_pipeline("PBR")
					.ok_or("PBR transparency pipeline not loaded!")?
					.clone();
				(pl, true, false)
			}
		};

		let mut cb = render_ctx.gather_commands(color_formats, Some(depth_format), None, viewport_extent)?;

		let pipeline_layout = pipeline.layout().clone();

		cb.bind_pipeline_graphics(pipeline.clone())?;

		if common_sets.len() > 0 {
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

			// look for any materials with transparency enabled or disabled (depending on `transparency_pass`)
			let continue_draw = original_materials
				.iter()
				.any(|(_, mat)| mat.has_transparency() == transparency_pass);

			if continue_draw {
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
					&mut cb,
					pipeline_layout.clone(),
					&projviewmodel,
					&mesh_resources.material_resources,
					transparency_pass,
					base_color_only,
					shadow_pass,
				)?;
			}
		}

		Ok(cb.build()?)
	}

	/// Free resources for the given entity ID. Only call this when the `Mesh` component was actually removed!
	pub fn cleanup_removed(&mut self, eid: EntityId)
	{
		self.resources.remove(&eid);
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

// Merge descriptor writes for the binding with variable descriptor count.
fn merge_descriptor_writes(writes: Vec<WriteDescriptorSet>) -> WriteDescriptorSet
{
	writes
		.into_iter()
		.reduce(|prev, cur| match prev.elements() {
			WriteDescriptorSetElements::ImageView(contents) => {
				let prev_contents = contents.iter().map(|img_info| img_info.image_view.clone());
				let cur_contents = match cur.elements() {
					WriteDescriptorSetElements::ImageView(c) => {
						c.iter().map(|img_info| img_info.image_view.clone())
					},
					_ => unreachable!(),
				};
				let elements = prev_contents.chain(cur_contents);
				WriteDescriptorSet::image_view_array(prev.binding(), 0, elements)
			}
			_ => unimplemented!(),
		})
		.unwrap()
}
