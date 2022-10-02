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
use crate::material::{ Material, DeferMaterialLoading };

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
		let parent_folder = path.parent().unwrap();

		// determine model file type
		match path.extension().and_then(|e| e.to_str()) {
			Some("glb") | Some("gltf") => {
				log::info!("Loading glTF file '{}'...", path.display());
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
			},
			Some("obj") => {
				log::info!("Loading OBJ file '{}'...", path.display());
				let (obj_models, obj_materials_result) = tobj::load_obj(&path, &tobj::GPU_LOAD_OPTIONS)?;
				let obj_materials = obj_materials_result?;
				
				Ok(Model{
					materials: obj_materials.iter()
						.map(|obj_mat| load_obj_mtl(&obj_mat, parent_folder, render_ctx))
						.collect::<Result<_, _>>()?,

					submeshes: obj_models.iter()
						.map(|obj_model| SubMesh::from_obj_mesh(render_ctx, &obj_model.mesh))
						.collect::<Result<_, _>>()?
				})
			},
			_ => Err(format!("couldn't determine model file type of {}", path.display()).into())
		}
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

fn load_obj_mtl(obj_mat: &tobj::Material, search_folder: &Path, render_ctx: &mut RenderContext)
	-> Result<Box<dyn Material>, GenericEngineError>
{
	let base_color = if obj_mat.diffuse_texture.is_empty() {
		crate::material::ColorInput::Color((Vec3::from(obj_mat.diffuse), obj_mat.dissolve).into())
	} else {
		crate::material::ColorInput::Texture(obj_mat.diffuse_texture.clone().into())
	};
	
	let mut loaded_mat = crate::material::pbr::PBR::new(base_color);
	loaded_mat.update_descriptor_set(search_folder, render_ctx)?;

	Ok(Box::new(loaded_mat))
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
	U32(Arc<BufferSlice<[u32], ImmutableBuffer<[u8]>>>),
	ObjU32(Arc<ImmutableBuffer<[u32]>>)
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
	pub fn from_obj_u32_vec(render_ctx: &mut RenderContext, indices: Vec<u32>)
		-> Result<Self, GenericEngineError>
	{
		Ok(Self::ObjU32(render_ctx.new_buffer_from_iter(indices, BufferUsage::index_buffer())?))
	}
	pub fn bind(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>)
	{
		match self {
			IndexBufferVariant::U16(buf) => cb.bind_index_buffer(buf.clone()),
			IndexBufferVariant::U32(buf) => cb.bind_index_buffer(buf.clone()),
			IndexBufferVariant::ObjU32(buf) => cb.bind_index_buffer(buf.clone())
		};
	}
}

struct SubMesh
{
	vertex_buffers: Vec<Arc<dyn BufferAccess>>,
	index_buf: IndexBufferVariant,
	vert_count: u32,
	mat_index: usize
}
impl SubMesh
{
	pub fn from_obj_mesh(render_ctx: &mut RenderContext, mesh: &tobj::Mesh)
		-> Result<Self, GenericEngineError>
	{
		if mesh.positions.is_empty() {
			return Err("no positions in OBJ mesh".into())
		}
		if mesh.texcoords.is_empty() {
			return Err("no texture coordinates in OBJ mesh".into())
		}
		if mesh.indices.is_empty() {
			return Err("no indices in OBJ mesh".into())
		}

		Ok(SubMesh{
			vertex_buffers: vec![
				render_ctx.new_buffer_from_iter(mesh.positions.clone(), BufferUsage::vertex_buffer())?,
				render_ctx.new_buffer_from_iter(mesh.texcoords.clone(), BufferUsage::vertex_buffer())?
			],
			index_buf: IndexBufferVariant::from_obj_u32_vec(render_ctx, mesh.indices.clone())?,
			vert_count: mesh.indices.len().try_into()?,
			mat_index: mesh.material_id.unwrap_or(0)
		})
	}

	pub fn from_gltf_primitive(prim: &gltf::Primitive, gpu_buffers: &Vec<Arc<ImmutableBuffer<[u8]>>>)
		-> Result<Self, GenericEngineError>
	{
		let positions = prim.get(&Semantic::Positions).ok_or("no positions in glTF primitive")?;
		let tex_coords = prim.get(&Semantic::TexCoords(0)).ok_or("no texture coordinates in glTF primitive")?;
		let indices = prim.indices().ok_or("no indices in glTF primitive")?;
		
		Ok(SubMesh{
			vertex_buffers: vec![
				get_buf_slice::<f32>(&positions, gpu_buffers)?,
				get_buf_slice::<f32>(&tex_coords, gpu_buffers)?
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

