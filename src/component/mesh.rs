/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use glam::*;
use vulkano::buffer::{ ImmutableBuffer, BufferUsage };
use vulkano::descriptor_set::{ WriteDescriptorSet, PersistentDescriptorSet };
use serde::Deserialize;
use crate::render::RenderContext;
use crate::component::DeferGpuResourceLoading;
use crate::component::EntityComponent;

#[derive(Deserialize)]
#[serde(from = "MeshData")]
pub struct Mesh
{
	pos_vert_buf: Option<Arc<ImmutableBuffer<[Vec3]>>>,
	uv_vert_buf: Option<Arc<ImmutableBuffer<[Vec2]>>>,
	index_buf: Option<Arc<ImmutableBuffer<[u32]>>>,
	mat_set: Option<Arc<PersistentDescriptorSet>>,
	
	data_to_load: Option<Box<MeshData>>
}
impl Mesh
{
	// TODO: set material
	/*pub fn new(render_ctx: &mut RenderContext, verts_pos: Vec<Vec3>, verts_uv: Vec<Vec2>, indices: Vec<u32>, color: Vec4)
		-> Result<Mesh, Box<dyn std::error::Error>>
	{
		let mat_buf = render_ctx.new_buffer_from_data(color, BufferUsage::uniform_buffer())?;
		let mat_set = render_ctx.new_descriptor_set("World", 2, [
			WriteDescriptorSet::buffer(0, mat_buf)
		])?;

		Ok(Mesh{
			pos_vert_buf: render_ctx.new_buffer_from_iter(pos_verts, BufferUsage::vertex_buffer())?,
			uv_vert_buf: render_ctx.new_buffer_from_iter(uv_verts, BufferUsage::vertex_buffer())?,
			index_buf: render_ctx.new_buffer_from_iter(indices, BufferUsage::index_buffer())?,
			mat_set: mat_set
		})
	}*/

	pub fn draw(&self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		render_ctx.bind_descriptor_set(2, self.mat_set.as_ref().ok_or("mesh not loaded")?.clone())?;
		render_ctx.bind_vertex_buffers(0, (
			self.pos_vert_buf.as_ref().ok_or("mesh not loaded")?.clone(), 
			self.uv_vert_buf.as_ref().ok_or("mesh not loaded")?.clone()
		));
		render_ctx.bind_index_buffers(self.index_buf.as_ref().ok_or("mesh not loaded")?.clone());
		render_ctx.draw(3, 1, 0, 0)?;
		Ok(())
	}
}
impl From<MeshData> for Mesh
{
	fn from(mesh_data: MeshData) -> Self 
	{
		Mesh{ 
			pos_vert_buf: None,
			uv_vert_buf: None,
			index_buf: None,
			mat_set: None,
			data_to_load: Some(Box::new(mesh_data)) 
		}
	}
}
impl DeferGpuResourceLoading for Mesh
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		if let Some(data) = self.data_to_load.take() {
			let mat_buf = render_ctx.new_buffer_from_data(data.color, BufferUsage::uniform_buffer())?;
			self.mat_set = Some(render_ctx.new_descriptor_set("World", 2, [
				WriteDescriptorSet::buffer(0, mat_buf)
			])?);

			self.pos_vert_buf = Some(render_ctx.new_buffer_from_iter(data.verts_pos, BufferUsage::vertex_buffer())?);
			self.uv_vert_buf = Some(render_ctx.new_buffer_from_iter(data.verts_uv, BufferUsage::vertex_buffer())?);
			self.index_buf = Some(render_ctx.new_buffer_from_iter(data.indices, BufferUsage::index_buffer())?);
		}
		Ok(())
	}
}
impl EntityComponent for Mesh
{}

#[derive(Deserialize)]
struct MeshData
{
	verts_pos: Vec<Vec3>,
	verts_uv: Vec<Vec2>,
	indices: Vec<u32>,
	color: Vec4
}


