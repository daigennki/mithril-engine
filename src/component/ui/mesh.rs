/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::buffer::ImmutableBuffer;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::command_buffer::DrawError;
use crate::rendercontext::texture::Texture;
use crate::rendercontext::RenderContext;

/// UI component that renders to a mesh, such as a quad, or a background frame mesh.
pub struct Mesh
{
	pos_vert_buf: Arc<ImmutableBuffer<[glam::Vec2]>>,
	uv_vert_buf: Arc<ImmutableBuffer<[glam::Vec2]>>,
	descriptor_set: Arc<PersistentDescriptorSet>,
	tex: Texture
}
impl Mesh
{
	pub fn new(render_ctx: &mut RenderContext, tex: Texture) -> Result<Mesh, Box<dyn std::error::Error>>
	{
		// create descriptor set
		let descriptor_set = render_ctx.new_ui_descriptor_set(1, [
			WriteDescriptorSet::image_view(0, tex.clone_view())
		])?;

		// vertex data
		let mut pos_verts: [glam::Vec2; 4] = [
			[ 0.0, 0.0 ].into(),
			[ 1.0, 0.0 ].into(),
			[ 0.0, 1.0 ].into(),
			[ 1.0, 1.0 ].into()
		];
		let uv_verts = pos_verts;

		// resize position vertices according to texture dimensions
		let dimensions_uvec2: glam::UVec2 = tex.dimensions().width_height().into();
		let dimensions = dimensions_uvec2.as_vec2();
		let half_dimensions = dimensions / 2.0;
		for pos in &mut pos_verts {
			*pos = pos.clone() * dimensions - half_dimensions;
		}

		Ok(Mesh{
			descriptor_set: descriptor_set,
			pos_vert_buf: render_ctx.new_buffer(pos_verts, BufferUsage::vertex_buffer())?,
			uv_vert_buf: render_ctx.new_buffer(uv_verts, BufferUsage::vertex_buffer())?,
			tex: tex
		})
	}

	pub fn tex_dimensions(&self) -> [u32; 2]
	{
		let dim = self.tex.dimensions();
		[ dim.width(), dim.height() ]
	}

	pub fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), DrawError>
	{
		render_ctx.bind_ui_descriptor_set(1, self.descriptor_set.clone());
		render_ctx.bind_vertex_buffers(0, (self.pos_vert_buf.clone(), self.uv_vert_buf.clone()));
		render_ctx.draw(4, 1, 0, 0)?;
		Ok(())
	}
}
