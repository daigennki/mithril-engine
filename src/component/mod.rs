/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod camera;
pub mod mesh;
pub mod ui;

use glam::*;
use serde::Deserialize;
use std::sync::{Arc, Mutex};
use vulkano::buffer::{allocator::SubbufferAllocator, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;

#[cfg(feature = "egui")]
use egui_winit_vulkano::egui;

use crate::render::RenderContext;
use crate::GenericEngineError;
use mithrilengine_derive::{EntityComponent, UniqueComponent};

#[derive(shipyard::Component, Deserialize, EntityComponent)]
pub struct Transform
{
	// TODO: parent-child relationship
	#[serde(skip)]
	staging_buf: Option<Mutex<SubbufferAllocator>>,
	#[serde(skip)]
	buf: Option<Subbuffer<Mat4>>,
	#[serde(skip)]
	descriptor_set: Option<Arc<PersistentDescriptorSet>>,

	position: Vec3,
	scale: Vec3,
	rotation: Vec3,
	is_static: Option<bool>,

	#[serde(skip)]
	rot_quat: Quat,
	#[serde(skip)]
	model_mat: Mat4,
}
impl Transform
{
	fn update_buffer(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		self.model_mat = Mat4::from_scale_rotation_translation(self.scale, self.rot_quat, self.position);
		let staged = self
			.staging_buf
			.as_ref()
			.ok_or("transform not loaded")?
			.lock()
			.or(Err("`Transform` staging buffer allocator mutex is poisoned!"))?
			.allocate_sized::<Mat4>()?;
		*staged.write()? = self.model_mat;
		render_ctx.copy_buffer(staged, self.buf.as_ref().ok_or("transform not loaded")?.clone())?;
		Ok(())
	}

	pub fn set_pos(&mut self, position: Vec3, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		self.position = position;
		self.update_buffer(render_ctx)
	}

	pub fn set_scale(&mut self, scale: Vec3, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		self.scale = scale;
		self.update_buffer(render_ctx)
	}

	/// Set the rotation of this object, in terms of X, Y, and Z axis rotations.
	pub fn set_rotation(&mut self, rotation: Vec3, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		self.rotation = rotation;
		let rot_rad = rotation * std::f32::consts::PI / 180.0;
		self.rot_quat = Quat::from_euler(EulerRot::XYZ, rot_rad.x, rot_rad.y, rot_rad.z);
		self.update_buffer(render_ctx)
	}

	pub fn is_this_static(&self) -> bool
	{
		self.is_static.unwrap_or(false)
	}
	pub fn position(&self) -> Vec3
	{
		self.position
	}
	pub fn scale(&self) -> Vec3
	{
		self.scale
	}
	pub fn rotation(&self) -> Vec3
	{
		self.rotation
	}
	pub fn rotation_quat(&self) -> &Quat
	{
		&self.rot_quat
	}
	pub fn get_matrix(&self) -> Mat4
	{
		self.model_mat
	}

	pub fn bind_descriptor_set(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
	) -> Result<(), GenericEngineError>
	{
		crate::render::bind_descriptor_set(cb, 0, self.descriptor_set.as_ref().ok_or("transform not loaded")?.clone())?;
		Ok(())
	}

	/// Show the egui collapsing header for this component.
	#[cfg(feature = "egui")]
	pub fn show_egui(&mut self, ui: &mut egui::Ui, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		let mut pos = self.position;
		egui::CollapsingHeader::new("Transform").show(ui, |ui| {
			if !self.is_this_static() {
				ui.columns(3, |cols| {
					cols[0].label("X");
					cols[0].add(egui::DragValue::new(&mut pos.x).speed(0.1));
					cols[1].label("Y");
					cols[1].add(egui::DragValue::new(&mut pos.y).speed(0.1));
					cols[2].label("Z");
					cols[2].add(egui::DragValue::new(&mut pos.z).speed(0.1));
				});
			}
		});
		self.set_pos(pos, render_ctx)?;

		Ok(())
	}
}
impl DeferGpuResourceLoading for Transform
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		let rot_rad = self.rotation * std::f32::consts::PI / 180.0;
		self.rot_quat = Quat::from_euler(EulerRot::XYZ, rot_rad.x, rot_rad.y, rot_rad.z);
		self.model_mat = Mat4::from_scale_rotation_translation(self.scale, self.rot_quat, self.position);

		let (staging_buf, buf) = render_ctx.new_cpu_buffer_from_data(self.model_mat, BufferUsage::UNIFORM_BUFFER)?;

		self.descriptor_set = Some(render_ctx.new_descriptor_set("PBR", 0, [WriteDescriptorSet::buffer(0, buf.clone())])?);
		self.staging_buf = Some(Mutex::new(staging_buf));
		self.buf = Some(buf);

		Ok(())
	}
}

#[typetag::deserialize]
pub trait EntityComponent: Send + Sync
{
	fn add_to_entity(self: Box<Self>, world: &mut shipyard::World, eid: shipyard::EntityId);
}
#[typetag::deserialize]
pub trait UniqueComponent: Send + Sync
{
	fn add_to_world(self: Box<Self>, world: &mut shipyard::World);
}

/// Trait for components which need `RenderContext` to finish loading their GPU resources after being deserialized.
pub trait DeferGpuResourceLoading
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>;
}

/*/// Trait for drawable components.
pub trait Draw
{
	fn draw(&self, command_buffer: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>;
}*/
