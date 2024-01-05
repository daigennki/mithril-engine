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
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderingInfo, CommandBufferUsage,
	PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo, SecondaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::{DescriptorSet, PersistentDescriptorSet};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearValue, Format};
use vulkano::image::view::ImageView;
use vulkano::pipeline::{
	graphics::viewport::Viewport,
	layout::{PipelineLayoutCreateInfo, PushConstantRange},
	GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

use crate::component::{EntityComponent, WantsSystemAdded};
use crate::material::MaterialPipelines;
use crate::render::{
	model::{ManagedModel, Model},
	RenderContext,
};

#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct Mesh
{
	pub model_path: PathBuf,

	#[serde(default)]
	pub material_variant: Option<String>,
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
#[derive(shipyard::Unique)]
pub struct MeshManager
{
	pipeline_layout: Arc<PipelineLayout>,
	pipeline_layout_oit: Arc<PipelineLayout>,

	material_pipelines: BTreeMap<&'static str, MaterialPipelines>,

	// Loaded 3D models, with the key being the path relative to the current working directory.
	// Each "managed model" will also contain the model instance for each entity.
	models: HashMap<PathBuf, ManagedModel>,

	// A mapping between entity IDs and the model it uses.
	resources: HashMap<EntityId, PathBuf>,

	cb_3d: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,
}
impl MeshManager
{
	pub fn new(render_ctx: &RenderContext) -> crate::Result<Self>
	{
		let vk_dev = render_ctx.device().clone();

		let material_textures_set_layout = render_ctx.get_material_textures_set_layout().clone();
		let light_set_layout = render_ctx.get_light_set_layout().clone();
		let transparency_input_layout = render_ctx.get_transparency_renderer().get_stage3_inputs().layout().clone();

		let push_constant_size = std::mem::size_of::<Mat4>() + std::mem::size_of::<Vec4>() * 3;
		let push_constant_range = PushConstantRange {
			stages: ShaderStages::VERTEX,
			offset: 0,
			size: push_constant_size.try_into().unwrap(),
		};
		let layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![material_textures_set_layout.clone(), light_set_layout.clone()],
			push_constant_ranges: vec![push_constant_range],
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(vk_dev.clone(), layout_info)?;

		let layout_info_oit = PipelineLayoutCreateInfo {
			set_layouts: vec![material_textures_set_layout, light_set_layout, transparency_input_layout],
			push_constant_ranges: vec![push_constant_range],
			..Default::default()
		};
		let pipeline_layout_oit = PipelineLayout::new(vk_dev.clone(), layout_info_oit)?;

		Ok(Self {
			pipeline_layout,
			pipeline_layout_oit,
			material_pipelines: Default::default(),
			models: Default::default(),
			resources: Default::default(),
			cb_3d: Default::default(),
		})
	}

	/// Load the model for the given `Mesh`.
	fn load(&mut self, render_ctx: &mut RenderContext, eid: EntityId, component: &Mesh) -> crate::Result<()>
	{
		// Get a 3D model from `path`, relative to the current working directory.
		// This attempts loading if it hasn't been loaded into memory yet.
		let managed_model = match self.models.get_mut(&component.model_path) {
			Some(m) => m,
			None => {
				let new_model = Arc::new(Model::new(render_ctx, &component.model_path)?);
				let managed = ManagedModel::new(new_model);
				self.models.insert(component.model_path.clone(), managed);
				self.models.get_mut(&component.model_path).unwrap()
			}
		};

		// Go through all the materials, and load the pipelines they need if they aren't already loaded.
		for mat in managed_model.model().get_materials() {
			let mat_name = mat.material_name();
			if !self.material_pipelines.contains_key(mat_name) {
				let pipeline_config = mat.load_shaders(self.pipeline_layout.device().clone())?;
				let pipeline_data =
					pipeline_config.into_pipelines(self.pipeline_layout.clone(), self.pipeline_layout_oit.clone())?;

				self.material_pipelines.insert(mat_name, pipeline_data);
			}
		}

		managed_model.new_user(eid, component.material_variant.clone());
		self.resources.insert(eid, component.model_path.clone());

		Ok(())
	}

	fn set_model_matrix(&mut self, eid: EntityId, model_matrix: Mat4)
	{
		let path = self.resources.get(&eid).unwrap().as_path();
		self.models.get_mut(path).unwrap().set_model_matrix(eid, model_matrix);
	}

	/// Free resources for the given entity ID. Only call this when the `Mesh` component was actually removed!
	fn cleanup_removed(&mut self, eid: EntityId)
	{
		let path = self.resources.get(&eid).unwrap().as_path();
		self.models.get_mut(path).unwrap().cleanup(eid);
		self.resources.remove(&eid);
	}

	pub fn draw(
		&self,
		render_ctx: &RenderContext,
		projview: Mat4,
		pass_type: PassType,
		common_sets: &[Arc<PersistentDescriptorSet>],
	) -> crate::Result<Option<Arc<SecondaryAutoCommandBuffer>>>
	{
		let (depth_format, viewport_extent, shadow_pass) = match &pass_type {
			PassType::Shadow {
				format, viewport_extent, ..
			} => (*format, *viewport_extent, true),
			_ => (crate::render::MAIN_DEPTH_FORMAT, render_ctx.swapchain_dimensions(), false),
		};

		let rendering_inheritance = CommandBufferInheritanceRenderingInfo {
			color_attachment_formats: pass_type.render_color_formats(),
			depth_attachment_format: Some(depth_format),
			..Default::default()
		};
		let mut cb = AutoCommandBufferBuilder::secondary(
			render_ctx.command_buffer_allocator(),
			render_ctx.graphics_queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
			CommandBufferInheritanceInfo {
				render_pass: Some(rendering_inheritance.into()),
				..Default::default()
			},
		)?;

		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [viewport_extent[0] as f32, viewport_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};
		cb.set_viewport(0, [viewport].as_slice().into())?;

		let pipeline_override = pass_type.pipeline();
		let transparency_pass = pass_type.transparency_pass();

		let mut any_drawn = false;
		for (pipeline_name, mat_pl) in &self.material_pipelines {
			let pipeline = if let Some(pl) = pipeline_override {
				pl.clone()
			} else if transparency_pass {
				if let Some(pl) = mat_pl.oit_pipeline.clone() {
					pl
				} else {
					continue;
				}
			} else {
				mat_pl.opaque_pipeline.clone()
			};

			cb.bind_pipeline_graphics(pipeline.clone())?;
			let pipeline_layout = pipeline.layout().clone();

			if common_sets.len() > 0 {
				let sets = Vec::from(common_sets);
				cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 1, sets)?;
			}

			// don't filter by material pipeline name if there is a pipeline override
			let pipeline_name_option = pipeline_override.is_none().then_some(pipeline_name);

			for managed_model in self.models.values() {
				if managed_model.draw(
					&mut cb,
					pipeline_name_option.copied(),
					pipeline_layout.clone(),
					transparency_pass,
					shadow_pass,
					&projview,
				)? {
					any_drawn = true;
				}
			}

			if pipeline_override.is_some() {
				break;
			}
		}

		// Don't bother building the command buffer if this is the OIT pass and no models were drawn.
		let cb_return = if any_drawn || !transparency_pass || shadow_pass {
			Some(cb.build()?)
		} else {
			None
		};
		Ok(cb_return)
	}

	pub fn add_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		*self.cb_3d.lock().unwrap() = Some(cb);
	}

	pub fn execute_rendering(
		&mut self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		color_image: Arc<ImageView>,
		depth_image: Arc<ImageView>,
	) -> crate::Result<()>
	{
		let main_render_info = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(color_image)
			})],
			depth_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store, // order-independent transparency needs this to be `Store`
				clear_value: Some(ClearValue::Depth(1.0)),
				..RenderingAttachmentInfo::image_view(depth_image)
			}),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		};
		let secondary_cb = self.cb_3d.lock().unwrap().take().unwrap();
		cb_builder
			.begin_rendering(main_render_info)?
			.execute_commands(secondary_cb)?
			.end_rendering()?;

		Ok(())
	}
}

pub enum PassType
{
	Shadow
	{
		pipeline: Arc<GraphicsPipeline>,
		format: Format,
		viewport_extent: [u32; 2],
	},
	Opaque,
	TransparencyMoments(Arc<GraphicsPipeline>),
	Transparency,
}
impl PassType
{
	fn render_color_formats(&self) -> Vec<Option<Format>>
	{
		let formats: &'static [Format] = match self {
			PassType::Shadow { .. } => &[],
			PassType::Opaque => &[Format::R16G16B16A16_SFLOAT],
			PassType::TransparencyMoments(_) => &[Format::R32G32B32A32_SFLOAT, Format::R32_SFLOAT, Format::R32_SFLOAT],
			PassType::Transparency => &[Format::R16G16B16A16_SFLOAT, Format::R8_UNORM],
		};
		formats.iter().copied().map(|f| Some(f)).collect()
	}
	fn pipeline(&self) -> Option<&Arc<GraphicsPipeline>>
	{
		match self {
			PassType::Shadow { pipeline, .. } | PassType::TransparencyMoments(pipeline) => Some(pipeline),
			_ => None,
		}
	}
	fn transparency_pass(&self) -> bool
	{
		match self {
			PassType::TransparencyMoments(_) | PassType::Transparency => true,
			_ => false,
		}
	}
}
