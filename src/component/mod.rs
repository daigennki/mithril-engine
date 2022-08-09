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
use crate::render::{ RenderContext, CommandBuffer };
use serde::Deserialize;
use component_derive::EntityComponent;
use component_derive::UniqueComponent;

#[derive(shipyard::Component, Deserialize,  EntityComponent)]
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

	#[serde(skip)]
	rot_quat: Quat
}
impl Transform
{
	/*pub fn new(render_ctx: &mut RenderContext, position: Vec3, scale: Vec3, rotation: Vec3) 
		-> Result<Transform, Box<dyn std::error::Error>>
	{
		let rot_quat = Quat::from_euler(EulerRot::XYZ, rotation.x, rotation.y, rotation.z);
		let transform_mat = Mat4::from_scale_rotation_translation(scale, rot_quat, position);
		let buf = render_ctx.new_cpu_buffer_from_data(transform_mat, BufferUsage::uniform_buffer())?;

		Ok(Transform{ 
			buf: buf.clone(),
			descriptor_set: render_ctx.new_descriptor_set("World", 0, [
				WriteDescriptorSet::buffer(0, buf)
			])?, 
			position: position, 
			scale: scale,
			rot_quat: rot_quat
		})
	}*/
	

	fn update_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		*self.buf.as_ref().ok_or("transform not loaded")?.write()? = 
			Mat4::from_scale_rotation_translation(self.scale, self.rot_quat, self.position);
		Ok(())
	}

	pub fn set_pos(&mut self, position: Vec3) -> Result<(), Box<dyn std::error::Error>>
	{
		self.position = position;
		self.update_buffer()
	}
	
	pub fn set_scale(&mut self, scale: Vec3) -> Result<(), Box<dyn std::error::Error>>
	{
		self.scale = scale;
		self.update_buffer()
	}

	/// Set the rotation of this object, in terms of X, Y, and Z axis rotations.
	pub fn set_rotation(&mut self, rotation: Vec3) -> Result<(), Box<dyn std::error::Error>>
	{
		self.rotation = rotation;
		self.rot_quat = Quat::from_euler(EulerRot::XYZ, rotation.x, rotation.y, rotation.z);
		self.update_buffer()
	}

	pub fn bind_descriptor_set<L>(&self, cb: &mut CommandBuffer<L>) -> Result<(), Box<dyn std::error::Error>>
	{
		cb.bind_descriptor_set(0, self.descriptor_set.as_ref().ok_or("transform not loaded")?.clone())?;
		Ok(())
	}
}
impl DeferGpuResourceLoading for Transform
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		self.rot_quat = Quat::from_euler(EulerRot::XYZ, self.rotation.x, self.rotation.y, self.rotation.z);
		let transform_mat = Mat4::from_scale_rotation_translation(self.scale, self.rot_quat, self.position);
		let buf = render_ctx.new_cpu_buffer_from_data(transform_mat, BufferUsage::uniform_buffer())?;

		self.descriptor_set = Some(render_ctx.new_descriptor_set("World", 0, [
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
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>;
}

/// Trait for drawable components.
pub trait Draw
{
	fn draw<L>(&self, command_buffer: &mut CommandBuffer<L>) -> Result<(), Box<dyn std::error::Error>>;
}


