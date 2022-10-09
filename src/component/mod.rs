/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod ui;
pub mod mesh;
pub mod camera;

use std::sync::Arc;
use glam::*;
use vulkano::buffer::{ /*ImmutableBuffer,*/ BufferUsage, cpu_access::CpuAccessibleBuffer };
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use crate::render::{ RenderContext, command_buffer::CommandBuffer };
use serde::Deserialize;
use mithrilengine_derive::{ EntityComponent, UniqueComponent };
use crate::GenericEngineError;

#[derive(shipyard::Component, Deserialize, EntityComponent)]
pub struct Transform
{
	// TODO: parent-child relationship
	// TODO: maybe we should use immutable buffers but only for static objects...
	#[serde(skip)]
	buf: Option<Arc<CpuAccessibleBuffer<Mat4>>>,
	#[serde(skip)]
	descriptor_set: Option<Arc<PersistentDescriptorSet>>,

	position: Vec3,
	scale: Vec3,
	rotation: Vec3,
	is_static: Option<bool>,

	#[serde(skip)]
	rot_quat: Quat
}
impl Transform
{
	fn update_buffer(&mut self) -> Result<(), GenericEngineError>
	{
		*self.buf.as_ref().ok_or("transform not loaded")?.write()? = 
			Mat4::from_scale_rotation_translation(self.scale, self.rot_quat, self.position);
		Ok(())
	}

	pub fn set_pos(&mut self, position: Vec3) -> Result<(), GenericEngineError>
	{
		self.position = position;
		self.update_buffer()
	}
	
	pub fn set_scale(&mut self, scale: Vec3) -> Result<(), GenericEngineError>
	{
		self.scale = scale;
		self.update_buffer()
	}

	/// Set the rotation of this object, in terms of X, Y, and Z axis rotations.
	pub fn set_rotation(&mut self, rotation: Vec3) -> Result<(), GenericEngineError>
	{
		self.rotation = rotation;
		let rot_rad = rotation * std::f32::consts::PI / 180.0;
		self.rot_quat = Quat::from_euler(EulerRot::XYZ, rot_rad.x, rot_rad.y, rot_rad.z);
		self.update_buffer()
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

	pub fn bind_descriptor_set<L>(&self, cb: &mut CommandBuffer<L>) -> Result<(), GenericEngineError>
	{
		cb.bind_descriptor_set(0, self.descriptor_set.as_ref().ok_or("transform not loaded")?.clone())?;
		Ok(())
	}
}
impl DeferGpuResourceLoading for Transform
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		let rot_rad = self.rotation * std::f32::consts::PI / 180.0;
		self.rot_quat = Quat::from_euler(EulerRot::XYZ, rot_rad.x, rot_rad.y, rot_rad.z);
		let transform_mat = Mat4::from_scale_rotation_translation(self.scale, self.rot_quat, self.position);
		let buf = render_ctx.new_cpu_buffer_from_data(transform_mat, BufferUsage::uniform_buffer())?;

		self.descriptor_set = Some(render_ctx.new_descriptor_set("PBR", 0, [
			WriteDescriptorSet::buffer(0, buf.clone())
		])?);
		self.buf = Some(buf);

		Ok(())
	}
}

#[typetag::deserialize]
pub trait EntityComponent
{
	fn add_to_entity(self: Box<Self>, world: &mut shipyard::World, eid: shipyard::EntityId);
}
#[typetag::deserialize]
pub trait UniqueComponent
{
	fn add_to_world(self: Box<Self>, world: &mut shipyard::World);
}


/// Trait for components which need `RenderContext` to finish loading their GPU resources after being deserialized.
pub trait DeferGpuResourceLoading
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>;
}

/// Trait for drawable components.
pub trait Draw
{
	fn draw(&self, command_buffer: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>;
}


