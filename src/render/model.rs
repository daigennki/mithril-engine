 /* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::any::TypeId;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use glam::*;
use gltf::accessor::DataType;
use gltf::Semantic;
use vulkano::buffer::{ ImmutableBuffer, BufferUsage, BufferContents, BufferAccess, BufferSlice };
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use crate::render::{ RenderContext, command_buffer::CommandBuffer };
use crate::GenericEngineError;
use crate::material::Material;

/// 3D model
pub struct Model
{
	materials: Vec<Box<dyn Material>>,
	submeshes: Vec<SubMesh>,
}
impl Model 
{
	pub fn new(render_ctx: &mut RenderContext, path: &Path) -> Result<Self, GenericEngineError>
	{
		log::info!("Loading glTF file '{}'...", path.display());
		let parent_folder = path.parent().unwrap();
		let (doc, data_buffers, _) = gltf::import(&path)?;
		
		// Load each glTF binary buffer into an `ImmutableBuffer`, from which buffer slices will be created.
		// This reduces memory fragmentation and transfers between CPU and GPU.
		let gpu_buf_usage = BufferUsage{
			vertex_buffer: true,
			index_buffer: true,
			..BufferUsage::none()
		};
		let gpu_buffers = data_buffers.iter()
			.map(|data_buffer| render_ctx.new_buffer_from_iter(data_buffer.0.clone(), gpu_buf_usage))
			.collect::<Result<_, _>>()?;

		Ok(Model{
			materials: doc.materials()
				.map(|mat| load_gltf_material(&mat, parent_folder, render_ctx))
				.collect::<Result<_, _>>()?,

			submeshes: doc.nodes()
				.filter_map(|node| node.mesh().map(|mesh| mesh.primitives()))
				.flatten()
				.map(|prim| SubMesh::from_gltf_primitive(&prim, &gpu_buffers))
				.collect::<Result<_, _>>()?
		})
	}

	pub fn draw(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		for submesh in &self.submeshes {
			// it's okay that we use a panic function here, since the glTF loader validates the index for us 
			self.materials[submesh.material_index()].bind_descriptor_set(cb)?;
			submesh.draw(cb)?;
		}
		Ok(())
	}
}

fn load_gltf_material(mat: &gltf::Material, search_folder: &Path, render_ctx: &mut RenderContext)
	-> Result<Box<dyn Material>, GenericEngineError>
{
	let material_name = mat.name().ok_or("glTF mesh material has no name")?;
	let mat_path = search_folder.join(material_name).with_extension("yaml");

	log::info!("Loading material file '{}'...", mat_path.display());
	let mut deserialized_mat: Box<dyn Material> = serde_yaml::from_reader(File::open(&mat_path)?)?;
	deserialized_mat.update_descriptor_set(&mat_path, render_ctx)?;
	Ok(deserialized_mat)
}

enum IndexBufferVariant
{
	U16(Arc<BufferSlice<[u16], ImmutableBuffer<[u8]>>>),
	U32(Arc<BufferSlice<[u32], ImmutableBuffer<[u8]>>>)
}
impl IndexBufferVariant
{
	pub fn from_accessor(accessor: &gltf::Accessor, gpu_buffers: &Vec<Arc<ImmutableBuffer<[u8]>>>)
		-> Result<Self, GenericEngineError>
	{
		Ok(match accessor.data_type() {
			DataType::U16 => IndexBufferVariant::U16(get_buf_slice(&accessor, gpu_buffers)?),
			DataType::U32 => IndexBufferVariant::U32(get_buf_slice(&accessor, gpu_buffers)?),
			other => return Err(format!("expected u16 or u32 index buffer, got '{:?}'", other).into())
		})
	}
	pub fn bind(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>)
	{
		match self {
			IndexBufferVariant::U16(buf) => cb.bind_index_buffer(buf.clone()),
			IndexBufferVariant::U32(buf) => cb.bind_index_buffer(buf.clone()),
		};
	}
}

struct SubMesh
{
	vertex_buffers: Vec<Arc<BufferSlice<[f32], ImmutableBuffer<[u8]>>>>,
	index_buf: IndexBufferVariant,
	vert_count: u32,
	mat_index: usize
}
impl SubMesh
{
	pub fn from_gltf_primitive(prim: &gltf::Primitive, gpu_buffers: &Vec<Arc<ImmutableBuffer<[u8]>>>)
		-> Result<Self, GenericEngineError>
	{
		let positions = prim.get(&Semantic::Positions).ok_or("no positions in glTF primitive")?;
		let tex_coords = prim.get(&Semantic::TexCoords(0)).ok_or("no texture coordinates in glTF primitive")?;
		let indices = prim.indices().ok_or("no indices in glTF primitive")?;
		
		Ok(SubMesh{
			vertex_buffers: vec![
				get_buf_slice(&positions, gpu_buffers)?,
				get_buf_slice(&tex_coords, gpu_buffers)?
			],
			index_buf: IndexBufferVariant::from_accessor(&indices, gpu_buffers)?,
			vert_count: indices.count().try_into()?,
			mat_index: prim.material().index().unwrap_or(0)
		})
	}

	pub fn draw(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		cb.bind_vertex_buffers(0, self.vertex_buffers.clone());
		self.index_buf.bind(cb);
		cb.draw_indexed(self.vert_count, 1, 0, 0, 0)?;
		Ok(())
	}

	pub fn material_index(&self) -> usize
	{
		self.mat_index
	}
}

fn data_type_to_id(value: DataType) -> TypeId
{
	match value {
		DataType::I8 => TypeId::of::<i8>(),
		DataType::U8 => TypeId::of::<u8>(),
		DataType::I16 => TypeId::of::<i16>(),
		DataType::U16 => TypeId::of::<u16>(),
		DataType::U32 => TypeId::of::<u32>(),
		DataType::F32 => TypeId::of::<f32>(),
	}
}
fn get_buf_slice<T>(accessor: &gltf::Accessor, gpu_buffers: &Vec<Arc<ImmutableBuffer<[u8]>>>)
	-> Result<Arc<BufferSlice<[T], ImmutableBuffer<[u8]>>>, GenericEngineError>
	where [T]: BufferContents
{
	if TypeId::of::<T>() != data_type_to_id(accessor.data_type()) {
		return Err(format!(
			"expected '{:?}', but given glTF primitive has `{:?}`", 
			TypeId::of::<T>(), 
			accessor.data_type()
		).into())
	}

	let view = accessor.view().ok_or("unexpected sparse accessor in glTF file")?;
	if view.stride().is_some() {
		return Err("unexpected interleaved data in glTF file".into())
	}
	let start = view.offset() as u64;
	let end = start + view.length() as u64;
	let buf_i = view.buffer().index();
	// The offset and length should've been validated by the glTF loader, 
	// hence why we use functions that may panic here.
	let buf_u8 = gpu_buffers[buf_i].slice(start..end).expect("buffer slice out of range");

	// This is a valid conversion as long as the glTF offsets and lengths are 
	// valid, which they should be since the glTF loader validates it for us.
	Ok(unsafe { buf_u8.reinterpret::<[T]>() })
}

