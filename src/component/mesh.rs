/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::path::{ Path, PathBuf };
use glam::*;
use gltf::accessor::DataType;
use serde::Deserialize;
use vulkano::buffer::{ 
	ImmutableBuffer, BufferUsage, BufferContents, BufferAccess, BufferSlice, 
	immutable::ImmutableBufferCreationError 
};
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
	#[serde(skip)]
	gpu_buffers: Vec<Arc<ImmutableBuffer<[u8]>>>,

	model_path: PathBuf
}
impl DeferGpuResourceLoading for Mesh
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		// model path relative to current directory
		let model_path_cd_rel = Path::new("./models/").join(&self.model_path);

		log::info!("Loading glTF file '{}'...", model_path_cd_rel.display());
		let (doc, mut data_buffers, _) = gltf::import(&model_path_cd_rel)?;
		
		// Load each glTF binary buffer into an `ImmutableBuffer`, from which buffer slices will be created.
		// This reduces memory fragmentation and transfers between CPU and GPU.
		let gpu_buf_usage = BufferUsage{
			vertex_buffer: true,
			index_buffer: true,
			..BufferUsage::none()
		};
		self.gpu_buffers.reserve(data_buffers.len());
		for data_buffer in data_buffers {
			self.gpu_buffers.push(render_ctx.new_buffer_from_iter(data_buffer.0, gpu_buf_usage)?);
		}

		for node in doc.nodes() {
			if let Some(mesh) = node.mesh() {
				for prim in mesh.primitives() {
					self.submeshes.push(
						SubMesh::from_gltf_primitive(prim, &model_path_cd_rel, &self.gpu_buffers, render_ctx)?
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

enum IndexBufferVariant
{
	U16(Arc<BufferSlice<[u16], ImmutableBuffer<[u8]>>>),
	U32(Arc<BufferSlice<[u32], ImmutableBuffer<[u8]>>>)
}
struct SubMesh
{
	vertex_buffers: Vec<Arc<dyn BufferAccess>>,
	index_buf: IndexBufferVariant,
	vert_count: u32,
	material: PBR 	// Box<dyn Material> doesn't compile for some reason, so we use this for now
}
impl SubMesh
{
	pub fn from_gltf_primitive(
		prim: gltf::Primitive, model_path: &Path, gpu_buffers: &Vec<Arc<ImmutableBuffer<[u8]>>>, render_ctx: &mut RenderContext
	)
		-> Result<Self, GenericEngineError>
	{
		let positions = prim
			.get(&gltf::Semantic::Positions)
			.ok_or("no positions in glTF primitive")?;
		let tex_coords = prim
			.get(&gltf::Semantic::TexCoords(0))
			.ok_or("no texture coordinates in glTF primitive")?;
		let indices = prim
			.indices()
			.ok_or("no indices in glTF primitive")?;
		let vert_count = indices.count();
		
		let vertex_buffers = vec![
			get_buf_slice_from_accessor::<f32>(&positions, gpu_buffers)? as Arc<dyn BufferAccess>,	// positions
			get_buf_slice_from_accessor::<f32>(&tex_coords, gpu_buffers)? as Arc<dyn BufferAccess>	// texture coordinates
		];
		let index_buf = match indices.data_type() {
			DataType::U16 => Ok(IndexBufferVariant::U16(get_buf_slice_from_accessor::<u16>(&indices, gpu_buffers)?)),
			DataType::U32 => Ok(IndexBufferVariant::U32(get_buf_slice_from_accessor::<u32>(&indices, gpu_buffers)?)),
			_ => Err(format!("expected u16 or u32 index buffer, got '{:?}'", indices.data_type()))
		}?;
		
		let mat_path = model_path.parent()
			.unwrap_or(Path::new("./models/"))
			.join(prim.material().name().ok_or("glTF mesh material has no name")?)
			.with_extension("yaml");
		log::info!("Loading material file '{}'...", mat_path.display());
		let mat_yaml_string = String::from_utf8(std::fs::read(mat_path)?)?;
		let mut deserialized_mat: PBR = serde_yaml::from_str(&mat_yaml_string)?;
		deserialized_mat.update_descriptor_set(render_ctx)?;

		Ok(SubMesh{
			vertex_buffers: vertex_buffers,
			index_buf: index_buf,
			vert_count: vert_count.try_into()?,
			material: deserialized_mat
		})
	}
	pub fn draw(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		self.material.bind_descriptor_set(cb)?;
		cb.bind_vertex_buffers(0, self.vertex_buffers.clone());
		match &self.index_buf {
			IndexBufferVariant::U16(buf) => cb.bind_index_buffer(buf.clone()),
			IndexBufferVariant::U32(buf) => cb.bind_index_buffer(buf.clone()),
		}
		cb.draw_indexed(self.vert_count, 1, 0, 0, 0)?;
		Ok(())
	}
}

fn get_buf_slice_from_accessor<T>(accessor: &gltf::Accessor, gpu_buffers: &Vec<Arc<ImmutableBuffer<[u8]>>>)
	-> Result<Arc<BufferSlice<[T], ImmutableBuffer<[u8]>>>, GenericEngineError>
	where [T]: BufferContents
{
	let view = accessor.view().ok_or("unexpected sparse accessor in glTF file")?;

	let start = view.offset() as u64;
	let end = start + view.length() as u64;
	let buf_u8 = gpu_buffers[view.buffer().index()]
		.slice(start..end)
		.ok_or(format!("slice between {} and {} bytes into vertex/index buffer is out of range", start, end))?;

	if std::any::TypeId::of::<T>() != match accessor.data_type() {
		DataType::I8 => std::any::TypeId::of::<i8>(),
		DataType::U8 => std::any::TypeId::of::<u8>(),
		DataType::I16 => std::any::TypeId::of::<i16>(),
		DataType::U16 => std::any::TypeId::of::<u16>(),
		DataType::U32 => std::any::TypeId::of::<u32>(),
		DataType::F32 => std::any::TypeId::of::<f32>(),
	} {
		return Err(format!("expected glTF data type '{:?}', got `{:?}`", std::any::TypeId::of::<T>(), accessor.data_type()).into())
	}

	Ok(unsafe { buf_u8.reinterpret::<[T]>() })
}

