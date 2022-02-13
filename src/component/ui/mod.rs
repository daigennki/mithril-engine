/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod mesh;
pub mod img;
pub mod canvas;

use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::command_buffer::DrawError;
use crate::rendercontext::RenderContext;

/// Common trait for UI elements.
pub trait UIElement
{
	fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), DrawError>;
}

pub struct Transform
{
	// TODO: parent-child relationship
	descriptor_set: Arc<PersistentDescriptorSet>
}
impl Transform
{
	pub fn new(render_ctx: &mut RenderContext, pos: glam::IVec2, scale: glam::Vec2, proj: glam::Mat4)
		-> Result<Transform, Box<dyn std::error::Error>>
	{
		let transformation = proj * glam::Mat4::from_scale_rotation_translation(
			scale.extend(0.0), 
			glam::Quat::IDENTITY, 
			pos.as_vec2().extend(0.0)
		);
		let transform_buf = render_ctx.new_buffer([transformation], BufferUsage::uniform_buffer())?;

		// create descriptor set
		let set_layout = render_ctx.get_ui_set_layout(0);
		let descriptor_set = PersistentDescriptorSet::new(set_layout, [
			WriteDescriptorSet::buffer(0, transform_buf.clone())
		])?;

		Ok(Transform{
			descriptor_set: descriptor_set
		})
	}

	pub fn bind_descriptor_set(&self, render_ctx: &mut RenderContext)
	{
		render_ctx.bind_ui_descriptor_set(0, self.descriptor_set.clone());
	}
}
