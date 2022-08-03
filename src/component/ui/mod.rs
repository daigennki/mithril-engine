/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod mesh;
pub mod canvas;
pub mod text;

use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use glam::*;
use crate::render::RenderContext;

pub struct Transform
{
	// TODO: parent-child relationship
	descriptor_set: Option<Arc<PersistentDescriptorSet>>,
	proj: Option<Mat4>,
	pos: IVec2,
	scale: Vec2
}
impl Transform
{
	pub fn new(pos: IVec2, scale: Vec2) -> Transform
	{
		Transform{ descriptor_set: None, proj: None, pos: pos, scale: scale }
	}

	pub fn bind_descriptor_set(&self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		let descriptor_set_ref = self.descriptor_set.as_ref()
			.ok_or("ui::Transform descriptor set bound before it was set up!")?;
		render_ctx.bind_descriptor_set(0, descriptor_set_ref.clone())?;
		Ok(())
	}

	pub fn update_projection(&mut self, render_ctx: &mut RenderContext, proj: Mat4)
		-> Result<(), Box<dyn std::error::Error>>
	{
		self.proj = Some(proj);
		self.descriptor_set = Some(update_matrix(render_ctx, proj, self.pos, self.scale)?);
		Ok(())
	}
}

fn update_matrix(render_ctx: &mut RenderContext, proj: Mat4, pos: IVec2, scale: Vec2) 
	-> Result<Arc<PersistentDescriptorSet>, Box<dyn std::error::Error>>
{
	let projected = proj * Mat4::from_scale_rotation_translation(
		scale.extend(0.0), 
		Quat::IDENTITY, 
		pos.as_vec2().extend(0.0)
	);
	let buf = render_ctx.new_buffer_from_data(projected, BufferUsage::uniform_buffer())?;

	// create descriptor set
	render_ctx.new_descriptor_set("UI", 0, [
		WriteDescriptorSet::buffer(0, buf.clone())
	])
}

/// Convenience function: create a tuple of `Transform` and `Mesh` to display an image loaded from a file on the UI.
pub fn new_image(render_ctx: &mut RenderContext, path: &str, pos: IVec2) 
	-> Result<(Transform, mesh::Mesh), Box<dyn std::error::Error>>
{
	let img_transform = Transform::new(pos, [ 1.0, 1.0 ].into());
	let img_tex = render_ctx.new_texture(std::path::Path::new(path))?;
	let img_mesh = mesh::Mesh::new(render_ctx, img_tex)?;

	Ok((img_transform, img_mesh))
}

/// Convenience function: create a tuple of `Transform` and `Text` to display text.
pub fn new_text(render_ctx: &mut RenderContext, text_str: &str, size: f32, pos: IVec2) 
	-> Result<(Transform, text::Text), Box<dyn std::error::Error>>
{
	let text_transform = Transform::new(pos, [ 1.0, 1.0 ].into());
	let text_mesh = text::Text::new(render_ctx, text_str, size)?;

	Ok((text_transform, text_mesh))
}
