/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod mesh;
pub mod canvas;

use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::rendercontext::RenderContext;

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


/// Convenience function: create a tuple of `Transform` and `Mesh` to display an image loaded from a file on the UI.
pub fn new_image(render_ctx: &mut RenderContext, canvas: &canvas::Canvas, path: &str, pos: glam::IVec2) 
	-> Result<(Transform, mesh::Mesh), Box<dyn std::error::Error>>
{
	let img_transform = Transform::new(render_ctx, pos, [ 1.0, 1.0 ].into(), canvas.projection())?;
	let img_tex = render_ctx.new_texture(std::path::Path::new(path))?;
	let img_mesh = mesh::Mesh::new(render_ctx, img_tex)?;

	Ok((img_transform, img_mesh))
}
