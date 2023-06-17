/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod canvas;
pub mod mesh;
pub mod text;

use glam::*;
use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;

use crate::render::RenderContext;
use crate::GenericEngineError;

#[derive(shipyard::Component)]
#[track(Insertion)]
pub struct Transform
{
	// TODO: parent-child relationship
	descriptor_set: Option<Arc<PersistentDescriptorSet>>,
	proj: Option<Mat4>,
	pos: IVec2,
	scale: Vec2,
}
impl Transform
{
	pub fn new(pos: IVec2, scale: Vec2) -> Self
	{
		Transform {
			descriptor_set: None,
			proj: None,
			pos,
			scale,
		}
	}

	pub fn bind_descriptor_set(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
	) -> Result<(), GenericEngineError>
	{
		let descriptor_set_ref = self
			.descriptor_set
			.as_ref()
			.ok_or("ui::Transform descriptor set bound before it was set up!")?;
		crate::render::bind_descriptor_set(cb, 0, descriptor_set_ref.clone())?;
		Ok(())
	}

	pub fn update_projection(&mut self, render_ctx: &mut RenderContext, proj: Mat4) -> Result<(), GenericEngineError>
	{
		self.proj = Some(proj);
		self.descriptor_set = Some(update_matrix(render_ctx, proj, self.pos, self.scale)?);
		Ok(())
	}
}

fn update_matrix(
	render_ctx: &mut RenderContext,
	proj: Mat4,
	pos: IVec2,
	scale: Vec2,
) -> Result<Arc<PersistentDescriptorSet>, GenericEngineError>
{
	let projected = proj * Mat4::from_scale_rotation_translation(scale.extend(0.0), Quat::IDENTITY, pos.as_vec2().extend(0.0));
	let buf = render_ctx.new_immutable_buffer_from_data(projected.to_cols_array(), BufferUsage::UNIFORM_BUFFER)?;

	// create descriptor set
	render_ctx.new_descriptor_set("UI", 0, [WriteDescriptorSet::buffer(0, buf.clone())])
}

/// Convenience function: create a tuple of `Transform` and `Mesh` to display an image loaded from a file on the UI.
pub fn new_image(render_ctx: &mut RenderContext, path: &str, pos: IVec2)
	-> Result<(Transform, mesh::Mesh), GenericEngineError>
{
	let img_transform = Transform::new(pos, Vec2::new(1.0, 1.0));
	let img_tex = render_ctx.get_texture(std::path::Path::new(path))?;
	let img_mesh = mesh::Mesh::new(render_ctx, img_tex)?;

	Ok((img_transform, img_mesh))
}

/// Convenience function: create a tuple of `Transform` and `Text` to display text.
pub fn new_text(
	render_ctx: &mut RenderContext,
	text: String,
	size: f32,
	pos: IVec2,
) -> Result<(Transform, text::Text), GenericEngineError>
{
	let text_transform = Transform::new(pos, Vec2::new(1.0, 1.0));
	let text_mesh = text::Text::new(render_ctx, text, size)?;

	Ok((text_transform, text_mesh))
}
