/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use glam::*;
use crate::render::RenderContext;
use crate::component::{ Transform, mesh::Mesh };

/// Create a tuple of `Transform` and `Mesh` to display a simple triangle.
pub fn new_triangle(render_ctx: &mut RenderContext, pos: Vec3, scale: Vec3, rot: Vec3, color: Vec4)
	-> Result<(Transform, Mesh), Box<dyn std::error::Error>>
{
	let tri_transform = Transform::new(render_ctx, pos, scale, rot)?;
	let tri_mesh = Mesh::new(render_ctx, color)?;

	Ok((tri_transform, tri_mesh))
}

