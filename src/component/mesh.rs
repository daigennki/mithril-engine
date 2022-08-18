/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::path::{ Path, PathBuf };
use glam::*;
use serde::Deserialize;
use vulkano::buffer::{ ImmutableBuffer, BufferUsage };
use vulkano::descriptor_set::{ WriteDescriptorSet, PersistentDescriptorSet };
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use crate::render::{ RenderContext, command_buffer::CommandBuffer };
use crate::component::{ EntityComponent, DeferGpuResourceLoading, Draw };
use crate::GenericEngineError;
use crate::material::Material;
use crate::material::pbr::PBR;

#[derive(shipyard::Component, Deserialize, EntityComponent)]
pub struct Mesh
{
	#[serde(skip)]
	submeshes: Vec<SubMesh>,

	model_path: PathBuf
}
impl DeferGpuResourceLoading for Mesh
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		/*if let Some(data) = self.data_to_load.take() {
			let mat_buf = render_ctx.new_buffer_from_data(data.color, BufferUsage::uniform_buffer())?;
			self.mat_set = Some(render_ctx.new_descriptor_set("World", 2, [
				WriteDescriptorSet::buffer(0, mat_buf)
			])?);

			self.pos_vert_buf = Some(render_ctx.new_buffer_from_iter(data.verts_pos, BufferUsage::vertex_buffer())?);
			self.uv_vert_buf = Some(render_ctx.new_buffer_from_iter(data.verts_uv, BufferUsage::vertex_buffer())?);
			self.index_buf = Some(render_ctx.new_buffer_from_iter(data.indices, BufferUsage::index_buffer())?);
		}*/

		// model path relative to current directory
		let model_path_cd_rel = Path::new("./models/").join(&self.model_path);
		log::info!("Loading glTF file '{}'...", model_path_cd_rel.display());
		let gltf = gltf::Gltf::open(&model_path_cd_rel)?;
		let binary_slice = gltf.blob.as_ref().ok_or("binary blob not found in glTF file")?.as_slice();
		for node in gltf.nodes() {
			if let Some(mesh) = node.mesh() {
				let transform = match node.transform() {
					gltf::scene::Transform::Matrix{ matrix } => Mat4::from_cols_array_2d(&matrix),
					gltf::scene::Transform::Decomposed{ translation, rotation, scale } => {
						Mat4::from_scale_rotation_translation(scale.into(), Quat::from_array(rotation), translation.into())
					}
				};
				for prim in mesh.primitives() {
					self.submeshes.push(
						SubMesh::from_gltf_primitive(prim, &model_path_cd_rel, binary_slice, transform, render_ctx)?
					);
				}
			}
		}

		Ok(())
	}
}
impl Draw for Mesh
{
	fn draw(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		for submesh in &self.submeshes {
			submesh.draw(cb)?;
		}
		Ok(())
	}
}

struct SubMesh
{
	pos_vert_buf: Arc<ImmutableBuffer<[[f32; 3]]>>,
	uv_vert_buf: Arc<ImmutableBuffer<[[f32; 2]]>>,
	index_buf: Arc<ImmutableBuffer<[u32]>>,
	vert_count: u32,
	material: PBR 	// Box<dyn Material> doesn't compile for some reason, so we use this for now
}
impl SubMesh
{
	pub fn from_gltf_primitive(
		prim: gltf::Primitive, model_path: &Path, binary_slice: &[u8], transform: Mat4, render_ctx: &mut RenderContext
	)
		-> Result<Self, GenericEngineError>
	{
		// TODO: there's probably a way we can use just one vertex buffer (but not the index buffer)
		// for all submeshes, owned by the parent `Mesh`.
		let prim_reader = prim.reader(|_/*buf*/| {
			/*assert_eq!(buf.length(), binary_slice.len());
			log::info!("glTF buffer length: {} (binary blob length: {})", buf.length(), binary_slice.len());
			log::info!("glTF buffer source: {:?}", buf.source());*/
			Some(binary_slice)
		});
		let mut positions = prim_reader
			.read_positions()
			.ok_or("no positions in glTF primitive")?
			.map(|pos| transform.transform_point3(pos.into()).to_array());	// transform vertices relative to scene origin
		let tex_coords = prim_reader
			.read_tex_coords(0)
			.ok_or("no texture coordinates in glTF primitive")?
			.into_f32();
		let indices = prim_reader
			.read_indices()
			.ok_or("no indices in glTF primitive")?
			.into_u32();

		let vert_count = indices.len();

		let pos_vert_buf = render_ctx.new_buffer_from_iter(positions, BufferUsage::vertex_buffer())?;
		let uv_vert_buf = render_ctx.new_buffer_from_iter(tex_coords, BufferUsage::vertex_buffer())?;
		let index_buf = render_ctx.new_buffer_from_iter(indices, BufferUsage::index_buffer())?;
		
		let mat_path = model_path.parent()
				.unwrap_or(Path::new("./models/"))
				.join(prim.material().name().ok_or("glTF mesh material has no name")?)
				.with_extension("yaml");
		log::info!("Loading material file '{}'...", mat_path.display());
		let mat_yaml_string = String::from_utf8(std::fs::read(mat_path)?)?;
		let mut deserialized_mat: PBR = serde_yaml::from_str(&mat_yaml_string)?;
		deserialized_mat.update_descriptor_set(render_ctx)?;

		Ok(SubMesh{
			pos_vert_buf: pos_vert_buf,
			uv_vert_buf: uv_vert_buf,
			index_buf: index_buf,
			vert_count: vert_count.try_into()?,
			material: deserialized_mat
		})
	}
	pub fn draw(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		self.material.bind_descriptor_set(cb)?;
		cb.bind_vertex_buffers(0, (
			self.pos_vert_buf.clone(), 
			self.uv_vert_buf.clone()
		));
		cb.bind_index_buffers(self.index_buf.clone());
		cb.draw_indexed(self.vert_count, 1, 0, 0, 0)?;
		Ok(())
	}
}

