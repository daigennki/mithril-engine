/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use crate::material::{pbr::PBR, ColorInput, DeferMaterialLoading, Material};
use crate::render::{command_buffer::CommandBuffer, RenderContext};
use crate::GenericEngineError;
use glam::*;
use gltf::accessor::DataType;
use gltf::Semantic;
use std::any::TypeId;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::{BufferAccess, BufferContents, BufferSlice, BufferUsage, DeviceLocalBuffer};
use vulkano::command_buffer::SecondaryAutoCommandBuffer;

/// 3D model
pub struct Model
{
	materials: Vec<Box<dyn Material>>,
	submeshes: Vec<SubMesh>,
	vertex_buffers: Vec<Arc<DeviceLocalBuffer<[f32]>>>,
	index_buffer: Arc<DeviceLocalBuffer<[u32]>>
}
impl Model
{
	pub fn new(render_ctx: &mut RenderContext, path: &Path, use_embedded_materials: bool) -> Result<Self, GenericEngineError>
	{
		let parent_folder = path.parent().unwrap();

		// determine model file type
		match path.extension().and_then(|e| e.to_str()) {
			Some("glb") | Some("gltf") => {
				log::info!("Loading glTF file '{}'...", path.display());
				let (doc, data_buffers, _) = gltf::import(&path)?;

				let primitives = doc
					.nodes()
					.filter_map(|node| node.mesh().map(|mesh| mesh.primitives()))
					.flatten();

				// Collect all of the vertex data into a single buffer to reduce the number of binds.
				let mut first_index: u32 = 0;
				let mut vertex_offset: i32 = 0;
				let mut positions = Vec::new();
				let mut texcoords = Vec::new();
				let mut indices = Vec::new();
				let mut submeshes = Vec::new();
				for prim in primitives {
					let positions_accessor = prim
						.get(&Semantic::Positions)
						.ok_or("no positions in glTF primitive")?;
					let positions_slice = get_buf_data::<f32>(&positions_accessor, &data_buffers)?;
					positions.extend_from_slice(positions_slice);

					let texcoords_accessor = prim
						.get(&Semantic::TexCoords(0))
						.ok_or("no texture coordinates in glTF primitive")?;
					let texcoords_slice = get_buf_data::<f32>(&texcoords_accessor, &data_buffers)?;
					texcoords.extend_from_slice(texcoords_slice);

					let indices_accessor = prim
						.indices()
						.ok_or("no indices in glTF primitive")?;
					match indices_accessor.data_type() {
						DataType::U16 => {
							// Convert the u16 indices into u32.
							let indices_slice = get_buf_data::<u16>(&indices_accessor, &data_buffers)?;
							let indices_u32 = indices_slice.iter().map(|index| *index as u32);
							indices.extend(indices_u32);
						}
						DataType::U32 => {
							let indices_slice = get_buf_data::<u32>(&indices_accessor, &data_buffers)?;
							indices.extend_from_slice(indices_slice);
						}
						other => return Err(format!("expected u16 or u32 index buffer, got '{:?}'", other).into()),
					}

					submeshes.push(SubMesh::from_gltf_primitive(&prim, first_index, vertex_offset)?);

					first_index += indices_accessor.count() as u32;
					vertex_offset += (positions_slice.len() / 3) as i32;
				}

				let vert_buf_usage = BufferUsage { vertex_buffer: true, ..BufferUsage::empty() };
				let vbo_positions = render_ctx.new_buffer_from_iter(positions, vert_buf_usage)?;
				let vbo_texcoords = render_ctx.new_buffer_from_iter(texcoords, vert_buf_usage)?;

				let index_buf_usage = BufferUsage { index_buffer: true, ..BufferUsage::empty() };
				let index_buffer = render_ctx.new_buffer_from_iter(indices, index_buf_usage)?;

				Ok(Model {
					materials: doc
						.materials()
						.map(|mat| load_gltf_material(&mat, parent_folder, render_ctx, use_embedded_materials))
						.collect::<Result<_, _>>()?,
					submeshes,
					vertex_buffers: vec![ vbo_positions, vbo_texcoords ],
					index_buffer
				})
			}
			/*Some("obj") => {
				log::info!("Loading OBJ file '{}'...", path.display());
				let (obj_models, obj_materials_result) = tobj::load_obj(&path, &tobj::GPU_LOAD_OPTIONS)?;
				let obj_materials = obj_materials_result?;

				Ok(Model {
					materials: obj_materials
						.iter()
						.map(|obj_mat| load_obj_mtl(&obj_mat, parent_folder, render_ctx))
						.collect::<Result<_, _>>()?,

					submeshes: obj_models
						.iter()
						.map(|obj_model| SubMesh::from_obj_mesh(render_ctx, &obj_model.mesh))
						.collect::<Result<_, _>>()?,
				})
			}*/
			_ => Err(format!("couldn't determine model file type of {}", path.display()).into()),
		}
	}

	pub fn get_materials(&mut self) -> &mut Vec<Box<dyn Material>>
	{
		&mut self.materials
	}

	pub fn draw(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		cb.bind_vertex_buffers(0, self.vertex_buffers.clone());
		cb.bind_index_buffer(self.index_buffer.clone());
		for submesh in &self.submeshes {
			// it's okay that we use a panic function here, since the glTF loader validates the index for us
			self.materials[submesh.material_index()].bind_descriptor_set(cb)?;
			submesh.draw(cb)?;
		}
		Ok(())
	}
}

/*fn load_obj_mtl(
	obj_mat: &tobj::Material, search_folder: &Path, render_ctx: &mut RenderContext,
) -> Result<Box<dyn Material>, GenericEngineError>
{
	let base_color = if obj_mat.diffuse_texture.is_empty() {
		ColorInput::Color((Vec3::from(obj_mat.diffuse), obj_mat.dissolve).into())
	} else {
		ColorInput::Texture(obj_mat.diffuse_texture.clone().into())
	};

	let mut loaded_mat = PBR::new(base_color);
	loaded_mat.update_descriptor_set(search_folder, render_ctx)?;

	Ok(Box::new(loaded_mat))
}*/

fn load_gltf_material(
	mat: &gltf::Material, search_folder: &Path, render_ctx: &mut RenderContext, use_embedded: bool,
) -> Result<Box<dyn Material>, GenericEngineError>
{
	let material_name = mat.name().ok_or("glTF mesh material has no name")?;
	let mat_path = search_folder.join(material_name).with_extension("yaml");

	if use_embedded {
		let base_color = ColorInput::Color(Vec4::from(mat.pbr_metallic_roughness().base_color_factor()));
		let mut loaded_mat = PBR::new(base_color);
		loaded_mat.update_descriptor_set(search_folder, render_ctx)?;
		Ok(Box::new(loaded_mat))
	} else {
		log::info!("Loading material file '{}'...", mat_path.display());
		let mut deserialized_mat: Box<dyn Material> = serde_yaml::from_reader(File::open(&mat_path)?)?;
		deserialized_mat.update_descriptor_set(search_folder, render_ctx)?;
		Ok(deserialized_mat)
	}
}

/*enum IndexBufferVariant
{
	U16(Arc<BufferSlice<[u16], DeviceLocalBuffer<[u8]>>>),
	U32(Arc<BufferSlice<[u32], DeviceLocalBuffer<[u8]>>>),
	ObjU32(Arc<DeviceLocalBuffer<[u32]>>),
}
impl IndexBufferVariant
{
	pub fn from_accessor(
		accessor: &gltf::Accessor, gpu_buffers: &Vec<Arc<DeviceLocalBuffer<[u8]>>>,
	) -> Result<Self, GenericEngineError>
	{
		Ok(match accessor.data_type() {
			DataType::U16 => IndexBufferVariant::U16(get_buf_slice(&accessor, gpu_buffers)?),
			DataType::U32 => IndexBufferVariant::U32(get_buf_slice(&accessor, gpu_buffers)?),
			other => return Err(format!("expected u16 or u32 index buffer, got '{:?}'", other).into()),
		})
	}
	pub fn from_obj_u32_vec(render_ctx: &mut RenderContext, indices: Vec<u32>) -> Result<Self, GenericEngineError>
	{
		Ok(Self::ObjU32(render_ctx.new_buffer_from_iter(indices, BufferUsage { index_buffer: true, ..BufferUsage::empty() })?))
	}
	pub fn bind(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>)
	{
		match self {
			IndexBufferVariant::U16(buf) => cb.bind_index_buffer(buf.clone()),
			IndexBufferVariant::U32(buf) => cb.bind_index_buffer(buf.clone()),
			IndexBufferVariant::ObjU32(buf) => cb.bind_index_buffer(buf.clone())
		};
	}
}*/

struct SubMesh
{
	first_index: u32,
	index_count: u32,
	vertex_offset: i32,
	mat_index: usize,
}
impl SubMesh
{
	/*pub fn from_obj_mesh(render_ctx: &mut RenderContext, mesh: &tobj::Mesh) -> Result<Self, GenericEngineError>
	{
		if mesh.positions.is_empty() {
			return Err("no positions in OBJ mesh".into());
		}
		if mesh.texcoords.is_empty() {
			return Err("no texture coordinates in OBJ mesh".into());
		}
		if mesh.indices.is_empty() {
			return Err("no indices in OBJ mesh".into());
		}

		let vert_buf_usage = BufferUsage { vertex_buffer: true, ..BufferUsage::empty() };
		Ok(SubMesh {
			vertex_buffers: vec![
				render_ctx.new_buffer_from_iter(mesh.positions.clone(), vert_buf_usage)?,
				render_ctx.new_buffer_from_iter(mesh.texcoords.clone(), vert_buf_usage)?,
			],
			index_buf: IndexBufferVariant::from_obj_u32_vec(render_ctx, mesh.indices.clone())?,
			vert_count: mesh.indices.len().try_into()?,
			first_vert_index: 0,
			mat_index: mesh.material_id.unwrap_or(0),
		})
	}*/

	pub fn from_gltf_primitive(
		prim: &gltf::Primitive, first_index: u32, vertex_offset: i32,
	) -> Result<Self, GenericEngineError>
	{
		let indices = prim.indices().ok_or("no indices in glTF primitive")?;

		Ok(SubMesh {
			first_index,
			index_count: indices.count().try_into()?,
			vertex_offset,
			mat_index: prim.material().index().unwrap_or(0),
		})
	}

	pub fn draw(&self, cb: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
	{
		cb.draw_indexed(self.index_count, 1, self.first_index, self.vertex_offset, 0)?;
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

/// Get a slice of the part of the buffer that the accessor points to.
fn get_buf_data<'a, T: 'static>(accessor: &gltf::Accessor, buffers: &'a Vec<gltf::buffer::Data>)
	-> Result<&'a [T], GenericEngineError>
{
	if TypeId::of::<T>() != data_type_to_id(accessor.data_type()) {
		return Err(
			format!("expected '{:?}', but given glTF primitive has `{:?}`", TypeId::of::<T>(), accessor.data_type()).into()
		);
	}

	let view = accessor
		.view()
		.ok_or("unexpected sparse accessor in glTF file")?;
	if view.stride().is_some() {
		return Err("unexpected interleaved data in glTF file".into());
	}
	let start = view.offset() as usize;
	let end = start + view.length() as usize;
	let buf_i = view.buffer().index();
	// The offset and length should've been validated by the glTF loader,
	// hence why we use functions that may panic here.
	let data_slice = &buffers[buf_i][start..end];
	let (_, reinterpreted_slice, _) = unsafe { data_slice.align_to::<T>() };
	Ok(reinterpreted_slice)
}

fn get_buf_slice<T>(
	accessor: &gltf::Accessor, gpu_buffers: &Vec<Arc<DeviceLocalBuffer<[u8]>>>,
) -> Result<Arc<BufferSlice<[T], DeviceLocalBuffer<[u8]>>>, GenericEngineError>
where
	[T]: BufferContents,
{
	if TypeId::of::<T>() != data_type_to_id(accessor.data_type()) {
		return Err(
			format!("expected '{:?}', but given glTF primitive has `{:?}`", TypeId::of::<T>(), accessor.data_type()).into()
		);
	}

	let view = accessor
		.view()
		.ok_or("unexpected sparse accessor in glTF file")?;
	if view.stride().is_some() {
		return Err("unexpected interleaved data in glTF file".into());
	}
	let start = view.offset() as u64;
	let end = start + view.length() as u64;
	let buf_i = view.buffer().index();
	// The offset and length should've been validated by the glTF loader,
	// hence why we use functions that may panic here.
	let buf_u8 = gpu_buffers[buf_i]
		.slice(start..end)
		.expect("buffer silce out of range");

	// This is a valid conversion as long as the glTF offsets and lengths are
	// valid, which they should be since the glTF loader validates it for us.
	Ok(unsafe { buf_u8.reinterpret::<[T]>() })
}
