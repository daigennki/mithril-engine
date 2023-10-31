/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use glam::*;
use gltf::accessor::DataType;
use gltf::Semantic;
use serde::Deserialize;
use std::any::TypeId;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::pipeline::{PipelineBindPoint, PipelineLayout};

use crate::material::{pbr::PBR, ColorInput, Material};
use crate::render::RenderContext;
use crate::GenericEngineError;

/// 3D model
pub struct Model
{
	materials: Vec<Box<dyn Material>>,
	submeshes: Vec<SubMesh>,
	vertex_buffers: Vec<Subbuffer<[f32]>>,
	index_buffer: IndexBufferVariant,
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

				// Collect all of the vertex data into buffers shared by all submeshes to reduce the number of binds.
				let mut first_index: u32 = 0;
				let mut vertex_offset: i32 = 0;
				let mut positions = Vec::new();
				let mut texcoords = Vec::new();
				let mut normals = Vec::new();
				let mut submeshes = Vec::new();
				let mut indices_u16 = Vec::new();
				let mut indices_u32 = Vec::new();

				let primitives = doc
					.nodes()
					.filter_map(|node| node.mesh().map(|mesh| mesh.primitives()))
					.flatten();
				for prim in primitives {
					let positions_accessor = prim.get(&Semantic::Positions).ok_or("no positions in glTF primitive")?;
					positions.extend_from_slice(get_buf_data(&positions_accessor, &data_buffers)?);

					let texcoords_accessor = prim
						.get(&Semantic::TexCoords(0))
						.ok_or("no texture coordinates in glTF primitive")?;
					texcoords.extend_from_slice(get_buf_data(&texcoords_accessor, &data_buffers)?);

					let normals_accessor = prim.get(&Semantic::Normals).ok_or("no normals in glTF primitive")?;
					normals.extend_from_slice(get_buf_data(&normals_accessor, &data_buffers)?);

					let indices_accessor = prim.indices().ok_or("no indices in glTF primitive")?;
					match indices_accessor.data_type() {
						DataType::U16 => indices_u16.extend_from_slice(get_buf_data(&indices_accessor, &data_buffers)?),
						DataType::U32 => indices_u32.extend_from_slice(get_buf_data(&indices_accessor, &data_buffers)?),
						other => return Err(format!("expected u16 or u32 index buffer, got '{:?}'", other).into()),
					};

					submeshes.push(SubMesh::from_gltf_primitive(&prim, first_index, vertex_offset)?);

					first_index += indices_accessor.count() as u32;
					vertex_offset += positions_accessor.count() as i32;
				}

				let vert_buf_usage = BufferUsage::VERTEX_BUFFER;
				let vbo_positions = render_ctx.new_buffer(positions, vert_buf_usage)?;
				let vbo_texcoords = render_ctx.new_buffer(texcoords, vert_buf_usage)?;
				let vbo_normals = render_ctx.new_buffer(normals, vert_buf_usage)?;

				Ok(Model {
					materials: doc
						.materials()
						.map(|mat| load_gltf_material(&mat, parent_folder))
						.collect::<Result<_, _>>()?,
					submeshes,
					vertex_buffers: vec![vbo_positions, vbo_texcoords, vbo_normals],
					index_buffer: IndexBufferVariant::from_u16_and_u32(render_ctx, indices_u16, indices_u32)?,
				})
			}
			_ => Err(format!("couldn't determine model file type of {}", path.display()).into()),
		}
	}

	pub fn get_materials(&self) -> &Vec<Box<dyn Material>>
	{
		&self.materials
	}

	/// Draw this model. `transform` is the model/projection/view matrices multiplied for frustum culling.
	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		pipeline_layout: Arc<PipelineLayout>,
		transform: &Mat4,
		material_resources: &Vec<crate::component::mesh::MaterialResources>,
		transparency_pass: bool,
		base_color_only: bool,
	) -> Result<(), GenericEngineError>
	{
		// determine which submeshes are visible
		let mut visible_submeshes = self.submeshes.iter().filter(|submesh| submesh.cull(transform)).peekable();

		// don't even bother with vertex/index buffer binds if no submeshes are visible
		if visible_submeshes.peek().is_some() {
			cb.bind_vertex_buffers(0, self.vertex_buffers.clone())?;
			self.index_buffer.bind(cb)?;
			for submesh in visible_submeshes {
				// it's okay that we use a panic function here, since the glTF loader validates the index for us
				let mat_res = &material_resources[submesh.material_index()];
				let mat = mat_res.mat_override.as_ref().unwrap_or_else(|| &self.materials[submesh.material_index()]);

				if mat.has_transparency() == transparency_pass {
					let set = if base_color_only {
						mat_res.mat_basecolor_only_set.clone()
					} else {
						mat_res.mat_set.clone()
					};
					cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 0, set)?;
					submesh.draw(cb)?;
				}
			}
		}
		Ok(())
	}
}

#[derive(Deserialize)]
struct MaterialExtras
{
	#[serde(default)]
	external: i32,
}

fn load_gltf_material(mat: &gltf::Material, search_folder: &Path) -> Result<Box<dyn Material>, GenericEngineError>
{
	// Use an external material file if specified in the extras.
	// This can be specified in Blender by giving a material a custom property called "external" with an integer value of 1.
	let use_external = if let Some(extras) = mat.extras() {
		let parsed_extras: MaterialExtras = serde_json::from_str(extras.get())?;
		parsed_extras.external != 0
	} else {
		false
	};

	if use_external {
		let material_name = mat.name().ok_or("model wants an external material, but the glTF mesh material has no name")?;
		let mat_path = search_folder.join(material_name).with_extension("yaml");

		log::info!(
			"External material specified, loading material file '{}'...",
			mat_path.display()
		);

		let deserialized_mat: Box<dyn Material> = serde_yaml::from_reader(File::open(&mat_path)?)?;

		Ok(deserialized_mat)
	} else {
		let transparent = mat.alpha_mode() == gltf::material::AlphaMode::Blend;
		if transparent {
			log::debug!("found a transparent glTF material");
		}

		let loaded_mat = PBR {
			base_color: ColorInput::Color(Vec4::from(mat.pbr_metallic_roughness().base_color_factor())),
			transparent,
		};

		Ok(Box::new(loaded_mat))
	}
}

enum IndexBufferVariant
{
	U16(Subbuffer<[u16]>),
	U32(Subbuffer<[u32]>),
}
impl IndexBufferVariant
{
	/// Determine the appropriate kind of index buffer, depending on whether the u16 or u32 index buffer is empty.
	/// If there are no u32 indices, a u16 index buffer will be created.
	/// If there are any u32 indices, a u32 index buffer will be created, and any u16 indices that may exist
	/// will be converted to u32.
	pub fn from_u16_and_u32(
		render_ctx: &mut RenderContext,
		indices_u16: Vec<u16>,
		mut indices_u32: Vec<u32>,
	) -> Result<Self, GenericEngineError>
	{
		// Convert the u16 indices into u32, if there are some u32 indices they will be mixed with.
		if !indices_u32.is_empty() && !indices_u16.is_empty() {
			let u16_to_u32 = indices_u16.iter().map(|index| *index as u32);
			indices_u32.extend(u16_to_u32);
		}

		let index_buf_usage = BufferUsage::INDEX_BUFFER;

		Ok(if indices_u32.is_empty() {
			IndexBufferVariant::U16(render_ctx.new_buffer(indices_u16, index_buf_usage)?)
		} else {
			IndexBufferVariant::U32(render_ctx.new_buffer(indices_u32, index_buf_usage)?)
		})
	}

	pub fn bind(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>)
		-> Result<(), GenericEngineError>
	{
		match self {
			IndexBufferVariant::U16(buf) => cb.bind_index_buffer(buf.clone()),
			IndexBufferVariant::U32(buf) => cb.bind_index_buffer(buf.clone()),
		}?;
		Ok(())
	}
}

struct SubMesh
{
	first_index: u32,
	index_count: u32,
	vertex_offset: i32,
	mat_index: usize,
	corner_min: Vec3,
	corner_max: Vec3,
}
impl SubMesh
{
	pub fn from_gltf_primitive(prim: &gltf::Primitive, first_index: u32, vertex_offset: i32)
		-> Result<Self, GenericEngineError>
	{
		let indices = prim.indices().ok_or("no indices in glTF primitive")?;

		Ok(SubMesh {
			first_index,
			index_count: indices.count().try_into()?,
			vertex_offset,
			mat_index: prim.material().index().unwrap_or(0),
			corner_min: prim.bounding_box().min.into(),
			corner_max: prim.bounding_box().max.into(),
		})
	}

	/// Perform frustum culling. Returns `true` if visible.
	pub fn cull(&self, projviewmodel: &Mat4) -> bool
	{
		// generate vertices for all 8 corners of this bounding box
		let mut bb_verts: [Vec4; 8] = Default::default();
		bb_verts[0..4].fill(self.corner_min.extend(1.0));
		bb_verts[4..8].fill(self.corner_max.extend(1.0));
		bb_verts[1].x = self.corner_max.x;
		bb_verts[2].x = self.corner_max.x;
		bb_verts[2].y = self.corner_max.y;
		bb_verts[3].y = self.corner_max.y;
		bb_verts[5].x = self.corner_min.x;
		bb_verts[6].x = self.corner_min.x;
		bb_verts[6].y = self.corner_min.y;
		bb_verts[7].y = self.corner_min.y;

		for vert in &mut bb_verts {
			*vert = *projviewmodel * *vert;
		}

		// check if all vertices are on the outside of any plane (evaluated in order of -X, +X, -Y, +Y, -Z, +Z)
		let mut neg_axis = true;
		for eval_axis in [0, 0, 1, 1, 2, 2] {
			// `outside` will only be true here if all vertices are outside of the plane being evaluated
			let outside = bb_verts.iter().all(|vert| {
				let mut axis_coord = vert[eval_axis];

				// negate coordinate when evaluating against negative planes so outside coordinates are greater than +W
				if neg_axis {
					axis_coord = -axis_coord;
				}

				axis_coord > vert.w // vertex is outside of plane if coordinate on plane axis is greater than +W
			});
			if outside {
				return false;
			}

			neg_axis = !neg_axis;
		}
		true
	}

	pub fn draw(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
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
fn get_buf_data<'a, T: 'static>(
	accessor: &gltf::Accessor,
	buffers: &'a Vec<gltf::buffer::Data>,
) -> Result<&'a [T], GenericEngineError>
{
	if TypeId::of::<T>() != data_type_to_id(accessor.data_type()) {
		return Err(format!(
			"expected '{:?}', but given glTF primitive has `{:?}`",
			TypeId::of::<T>(),
			accessor.data_type()
		)
		.into());
	}

	let view = accessor.view().ok_or("unexpected sparse accessor in glTF file")?;
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
