/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::any::TypeId;
use std::sync::Arc;
use std::path::{ Path, PathBuf };
use glam::*;
use gltf::accessor::DataType;
use serde::Deserialize;
use vulkano::buffer::{ ImmutableBuffer, BufferUsage, BufferContents, BufferAccess, BufferSlice };
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
	#[serde(skip)]
	materials: Vec<PBR>,	// Box<dyn Material> doesn't compile for some reason, so we use this for now

	model_path: PathBuf
}
impl DeferGpuResourceLoading for Mesh
{
	fn finish_loading(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>
	{
		// model path relative to current directory
		let model_path_cd_rel = Path::new("./models/").join(&self.model_path);

		log::info!("Loading glTF file '{}'...", model_path_cd_rel.display());
		let (doc, data_buffers, _) = gltf::import(&model_path_cd_rel)?;
		
		// Load each glTF binary buffer into an `ImmutableBuffer`, from which buffer slices will be created.
		// This reduces memory fragmentation and transfers between CPU and GPU.
		// It might also be loading some unnecessary data into the GPU though...
		let gpu_buf_usage = BufferUsage{
			vertex_buffer: true,
			index_buffer: true,
			..BufferUsage::none()
		};
		self.gpu_buffers.reserve(data_buffers.len());
		for data_buffer in data_buffers {
			self.gpu_buffers.push(render_ctx.new_buffer_from_iter(data_buffer.0, gpu_buf_usage)?);
		}

		self.materials.reserve(doc.materials().len());
		for mat in doc.materials() {
			let mat_path = model_path_cd_rel.parent().unwrap()
				.join(mat.name().ok_or("glTF mesh material has no name")?)
				.with_extension("yaml");

			log::info!("Loading material file '{}'...", mat_path.display());
			let mat_yaml_string = String::from_utf8(std::fs::read(mat_path)?)?;
			let mut deserialized_mat: PBR = serde_yaml::from_str(&mat_yaml_string)?;
			deserialized_mat.update_descriptor_set(render_ctx)?;
			self.materials.push(deserialized_mat);
		}

		for node in doc.nodes() {
			if let Some(mesh) = node.mesh() {
				for prim in mesh.primitives() {
					let submesh = SubMesh::from_gltf_primitive(prim, &model_path_cd_rel, &self.gpu_buffers, render_ctx)?;
					// make sure that the material index isn't out of bounds, so we don't have to do error checking before draw
					if submesh.material_index() >= self.materials.len() {
						return Err(format!("Material index {} for submesh is out of bounds!", submesh.material_index()).into())
					}
					self.submeshes.push(submesh);
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
			// it's okay that we use a panic function here, since we already checked the material indices earlier
			self.materials[submesh.material_index()].bind_descriptor_set(cb)?;
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
	vertex_buffers: Vec<Arc<BufferSlice<[f32], ImmutableBuffer<[u8]>>>>,
	index_buf: IndexBufferVariant,
	vert_count: u32,
	mat_index: usize
}
impl SubMesh
{
	pub fn from_gltf_primitive(
		prim: gltf::Primitive, model_path: &Path, gpu_buffers: &Vec<Arc<ImmutableBuffer<[u8]>>>, render_ctx: &mut RenderContext
	)
		-> Result<Self, GenericEngineError>
	{
		let positions = prim.get(&gltf::Semantic::Positions).ok_or("no positions in glTF primitive")?;
		let tex_coords = prim.get(&gltf::Semantic::TexCoords(0)).ok_or("no texture coordinates in glTF primitive")?;
		let vertex_buffers = vec![
			get_buf_slice::<f32>(&positions, gpu_buffers)?,
			get_buf_slice::<f32>(&tex_coords, gpu_buffers)?
		];

		let indices = prim.indices().ok_or("no indices in glTF primitive")?;
		let vert_count = indices.count();
		let index_buf = match indices.data_type() {
			DataType::U16 => Ok(IndexBufferVariant::U16(get_buf_slice::<u16>(&indices, gpu_buffers)?)),
			DataType::U32 => Ok(IndexBufferVariant::U32(get_buf_slice::<u32>(&indices, gpu_buffers)?)),
			_ => Err(format!("expected u16 or u32 index buffer, got '{:?}'", indices.data_type()))
		}?;
		
		Ok(SubMesh{
			vertex_buffers: vertex_buffers,
			index_buf: index_buf,
			vert_count: vert_count.try_into()?,
			mat_index: prim.material().index().unwrap_or(0)
		})
	}

	pub fn draw(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		cb.bind_vertex_buffers(0, self.vertex_buffers.clone());
		match &self.index_buf {
			IndexBufferVariant::U16(buf) => cb.bind_index_buffer(buf.clone()),
			IndexBufferVariant::U32(buf) => cb.bind_index_buffer(buf.clone()),
		}
		cb.draw_indexed(self.vert_count, 1, 0, 0, 0)?;
		Ok(())
	}

	pub fn material_index(&self) -> usize
	{
		self.mat_index
	}
}

fn get_buf_slice<T>(accessor: &gltf::Accessor, gpu_buffers: &Vec<Arc<ImmutableBuffer<[u8]>>>)
	-> Result<Arc<BufferSlice<[T], ImmutableBuffer<[u8]>>>, GenericEngineError>
	where [T]: BufferContents
{
	if TypeId::of::<T>() != match accessor.data_type() {
		DataType::I8 => TypeId::of::<i8>(),
		DataType::U8 => TypeId::of::<u8>(),
		DataType::I16 => TypeId::of::<i16>(),
		DataType::U16 => TypeId::of::<u16>(),
		DataType::U32 => TypeId::of::<u32>(),
		DataType::F32 => TypeId::of::<f32>(),
	} {
		return Err(format!(
			"expected '{:?}', but given glTF primitive has `{:?}`", 
			TypeId::of::<T>(), 
			accessor.data_type()
		).into())
	}

	let view = accessor.view().ok_or("unexpected sparse accessor in glTF file")?;
	let start = view.offset() as u64;
	let end = start + view.length() as u64;
	let buf_u8 = gpu_buffers[view.buffer().index()]
		.slice(start..end)
		.ok_or(format!("slice between {} and {} bytes into vertex/index buffer is out of range", start, end))?;

	// This should be a valid conversion as long as the glTF offsets and lengths are valid.
	// Maybe we should check alignment though, just in case?
	Ok(unsafe { buf_u8.reinterpret::<[T]>() })
}

