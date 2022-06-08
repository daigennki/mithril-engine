/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod ui;

use std::sync::Arc;
use glam::*;
use vulkano::buffer::{ ImmutableBuffer, BufferUsage };
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::command_buffer::DrawError;
use crate::render::texture::Texture;
use crate::render::RenderContext;

pub struct Transform
{
	// TODO: parent-child relationship
	descriptor_set: Option<Arc<PersistentDescriptorSet>>,
	proj: Option<Mat4>,
	pos: Vec3,
	scale: Vec3
}
impl Transform
{
	pub fn new(pos: Vec3, scale: Vec3) -> Transform
	{
		Transform{ descriptor_set: None, proj: None, pos: pos, scale: scale }
	}

	pub fn bind_descriptor_set(&self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		let descriptor_set_ref = self.descriptor_set.as_ref()
			.ok_or("ui::Transform descriptor set bound before it was set up!")?;
		render_ctx.bind_ui_descriptor_set(0, descriptor_set_ref.clone());
		Ok(())
	}

	/*pub fn update_projection(&mut self, render_ctx: &mut RenderContext, proj: Mat4)
		-> Result<(), Box<dyn std::error::Error>>
	{
		self.proj = Some(proj);
		self.descriptor_set = Some(update_matrix(render_ctx, proj, self.pos, self.scale)?);
		Ok(())
	}*/
}
