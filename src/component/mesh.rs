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
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout};

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

/// A single manager that manages the GPU resources for all `Mesh` components.
#[derive(Default, shipyard::Unique)]
pub struct MeshManager
{
	material_pipelines: HashMap<&'static str, MaterialPipelines>,

	// Loaded 3D models, with the key being the path relative to the current working directory.
	models: BTreeMap<PathBuf, Arc<Model>>,

	// The models and materials for each entity.
	resources: HashMap<EntityId, MeshResources>,

	cb_3d: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,
}
impl MeshManager
{
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
		
		self.resources.insert(eid, MeshResources::new(render_ctx, model_data)?);

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

		// go through all the loaded meshes
		for mesh_resources in self.resources.values() {
			mesh_resources.draw(&mut cb, pipeline_layout.clone(), transparency_pass, shadow_pass, &projview)?;
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

struct MeshResources
{
	model: Arc<Model>,

	// Image views for each material in use by the current material group.
	// Uses variable descriptor count.
	textures_set: Arc<PersistentDescriptorSet>,
	textures_count: u32, // The number of textures in the variable count descriptor.

	// The texture base indices in the variable descriptor count of `textures_set`.
	// (to get the texture base index, use the material index as an index to this)
	mat_tex_base_indices: Vec<u32>,

	model_matrix: Mat4,
}
impl MeshResources
{
	pub fn new(render_ctx: &mut RenderContext, model: Arc<Model>) -> Result<Self, GenericEngineError>
	{
		let parent_folder = model.path().parent().unwrap();
		let original_materials = model.get_materials();

		// Get the image views for each material, and calculate the base index in the variable descriptor count.
		let mut image_view_writes = Vec::with_capacity(original_materials.len());
		let mut mat_tex_base_indices = Vec::with_capacity(original_materials.len());
		for mat in original_materials {
			let mat_image_views = mat.gen_descriptor_set_write(parent_folder, render_ctx)?;
			image_view_writes.push(mat_image_views);

			let next_mat_tex_base_index = mat_tex_base_indices.last().copied().unwrap_or(0) + mat.tex_index_stride();
			mat_tex_base_indices.push(next_mat_tex_base_index);
		}
		// There will be one extra unused element at the end of `mat_tex_base_indices`, so remove it,
		// then make sure the first material has a base texture index of 0.
		mat_tex_base_indices.pop();
		mat_tex_base_indices.insert(0, 0);

		// Make a single write out of the image views of all of the materials, and create a single descriptor set.
		let variable_descriptor_count = image_view_writes.len().try_into()?;
		log::debug!("variable descriptor count: {}", variable_descriptor_count);

		let textures_set = PersistentDescriptorSet::new_variable(
			render_ctx.descriptor_set_allocator(),
			render_ctx.get_material_textures_set_layout().clone(),
			variable_descriptor_count,
			[WriteDescriptorSet::image_view_array(1, 0, image_view_writes.into_iter().flatten())],
			[]
		)?;

		Ok(MeshResources {
			model,
			textures_set,
			textures_count: variable_descriptor_count,
			mat_tex_base_indices,
			model_matrix: Default::default(),
		})
	}

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		pipeline_layout: Arc<PipelineLayout>,
		transparency_pass: bool,
		shadow_pass: bool,
		projview: &Mat4,
	) -> Result<(), GenericEngineError>
	{
		let materials = self.model.get_materials();

		// look for any materials with transparency enabled or disabled (depending on `transparency_pass`)
		let draw_this_mesh = materials
			.iter()
			.any(|mat| mat.has_transparency() == transparency_pass);
		if !draw_this_mesh {
			return Ok(()); // skip to the next mesh if none of the materials match this pass type
		}

		let projviewmodel = *projview * self.model_matrix;
		if shadow_pass {
			// TODO: also consider point lights, which require different matrices
			cb.push_constants(pipeline_layout, 0, projviewmodel)?;
		} else {
			let translation = self.model_matrix.w_axis.xyz();
			let push_data = MeshPushConstant {
				projviewmodel,
				model_x: self.model_matrix.x_axis.xyz().extend(translation.x),
				model_y: self.model_matrix.y_axis.xyz().extend(translation.y),
				model_z: self.model_matrix.z_axis.xyz().extend(translation.z),
			};
			cb.push_constants(pipeline_layout.clone(), 0, push_data)?;

			let set = self.textures_set.clone();
			cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout, 0, set)?;
		}

		if let Some(last_mat_tex_base_index) = self.mat_tex_base_indices.last() {
			let last_mat_stride = materials[materials.len() - 1].tex_index_stride();

			// Make sure that the shader doesn't overrun the variable count descriptor.
			// Some very weird things (like crashing the entire computer) might happen if we don't check this!
			assert!(last_mat_tex_base_index + last_mat_stride <= self.textures_count);
		}

		self.model.draw(cb, &projviewmodel, &self.mat_tex_base_indices, transparency_pass, shadow_pass)?;

		Ok(())
	}
}

struct MaterialPipelines
{
	opaque_pipeline: Arc<GraphicsPipeline>,
	oit_pipeline: Option<Arc<GraphicsPipeline>>, // Optional transparency pipeline may also be specified.
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

