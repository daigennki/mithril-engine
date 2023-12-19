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
use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{PipelineBindPoint, PipelineLayout};

use crate::material::{pbr::PBR, ColorInput, Material};
use crate::render::RenderContext;
use crate::GenericEngineError;

/// 3D model
pub struct Model
{
	materials: Vec<Box<dyn Material>>,
	material_variants: Vec<String>,
	submeshes: Vec<SubMesh>,
	vertex_subbuffers: Vec<Subbuffer<[f32]>>,
	index_buffer: IndexBufferVariant,
	path: PathBuf,

	// Image views for all materials used by all material variants the glTF document comes with.
	// Uses variable descriptor count.
	textures_set: Arc<PersistentDescriptorSet>,

	// The texture base indices in the variable descriptor count of `textures_set`.
	// (to get the texture base index, use the material index as an index to this)
	mat_tex_base_indices: Vec<u32>,
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
				let mut indices_type = DataType::U16;

				let materials: Vec<_> = doc
					.materials()
					.map(|mat| load_gltf_material(&mat, parent_folder))
					.collect::<Result<_, _>>()?;

				// Collect material variants. If there are no material variants, only one material
				// group will be set up in the end.
				let material_variants: Vec<_> = doc
					.variants()
					.into_iter()
					.flatten()
					.map(|variant| variant.name().to_string())
					.collect();

				if material_variants.is_empty() {
					log::debug!("no material variants in model");
				} else {
					log::debug!("material variants in model:");
					material_variants.iter().enumerate().for_each(|(i, variant)| log::debug!("{i}: {}", &variant));
				}

				// Create submeshes from glTF "primitives".
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
						DataType::U16 => {
							if indices_type == DataType::U32 {
								// convert the indices to u32 if mixed with u32 indices
								let slice_u16: &[u16] = get_buf_data(&indices_accessor, &data_buffers)?;
								indices_u32.extend(slice_u16.iter().map(|index| *index as u32));
							} else {
								indices_u16.extend_from_slice(get_buf_data(&indices_accessor, &data_buffers)?)
							}
						}
						DataType::U32 => {
							if indices_type == DataType::U16 {
								// convert existing indices to u32 if they're u16
								indices_type = DataType::U32;
								indices_u32 = indices_u16.drain(..).map(|index| index as u32).collect();
								indices_u16 = Vec::new(); // free some memory
							}

							indices_u32.extend_from_slice(get_buf_data(&indices_accessor, &data_buffers)?);
						}
						other => return Err(format!("expected u16 or u32 index buffer, got '{:?}'", other).into()),
					};

					let submesh = SubMesh::from_gltf_primitive(&prim, first_index, vertex_offset)?;

					submeshes.push(submesh);

					first_index += indices_accessor.count() as u32;
					vertex_offset += positions_accessor.count() as i32;
				}

				// Check what shaders are being used across all of the materials in this model.
				let mut shader_names = BTreeSet::new();
				for mat in &materials {
					if !shader_names.contains(mat.material_name()) {
						shader_names.insert(mat.material_name());
					}
				}
				log::debug!("Model has {} submeshes. It uses these shaders:", submeshes.len());
				shader_names.iter().for_each(|shader_name| log::debug!("- {shader_name}"));

				// Combine the vertex data into a single buffer,
				// then split it into subbuffers for different types of vertex data.
				let mut combined_data = Vec::with_capacity(positions.len() + texcoords.len() + normals.len());
				combined_data.append(&mut positions);
				let texcoords_offset: u64 = combined_data.len().try_into()?;
				combined_data.append(&mut texcoords);
				let normals_offset: u64 = combined_data.len().try_into()?;
				combined_data.append(&mut normals);

				let vert_buf_usage = BufferUsage::VERTEX_BUFFER;
				let vertex_buffer = render_ctx.new_buffer(combined_data.as_slice(), vert_buf_usage)?;
				let vbo_positions = vertex_buffer.clone().slice(..texcoords_offset);
				let vbo_texcoords = vertex_buffer.clone().slice(texcoords_offset..normals_offset);
				let vbo_normals = vertex_buffer.clone().slice(normals_offset..);

				let index_buf_usage = BufferUsage::INDEX_BUFFER;
				let index_buffer = match indices_type {
					DataType::U16 => IndexBufferVariant::U16(render_ctx.new_buffer(&indices_u16, index_buf_usage)?),
					DataType::U32 => IndexBufferVariant::U32(render_ctx.new_buffer(&indices_u32, index_buf_usage)?),
					_ => unreachable!(),
				};

				// Get the image views for each material, and calculate the base index in the variable descriptor count.
				let mut image_view_writes = Vec::with_capacity(materials.len());
				let mut mat_tex_base_indices = Vec::with_capacity(materials.len());
				let mut last_tex_index_stride = 0;
				for mat in &materials {
					let mat_image_views = mat.gen_descriptor_set_write(parent_folder, render_ctx)?;
					last_tex_index_stride = mat_image_views.len().try_into()?;
					image_view_writes.push(mat_image_views);

					let next_mat_tex_base_index = mat_tex_base_indices.last().copied().unwrap_or(0) + last_tex_index_stride;
					mat_tex_base_indices.push(next_mat_tex_base_index);
				}
				// There will be one extra unused element at the end of `mat_tex_base_indices`, so remove it,
				// then give the first material a base texture index of 0.
				mat_tex_base_indices.pop();
				mat_tex_base_indices.insert(0, 0);

				let texture_count = image_view_writes.iter().flatten().count().try_into()?;
				log::debug!("texture count (variable descriptor count): {}", texture_count);

				// Make sure that the shader doesn't overrun the variable count descriptor.
				// Some very weird things (like crashing the entire computer) might happen if we don't check this!
				let last_mat_tex_base_index = mat_tex_base_indices.last().unwrap();
				assert!(last_mat_tex_base_index + last_tex_index_stride <= texture_count);

				// Make a single write out of the image views of all of the materials, and create a single descriptor set.
				let textures_set = PersistentDescriptorSet::new_variable(
					render_ctx.descriptor_set_allocator(),
					render_ctx.get_material_textures_set_layout().clone(),
					texture_count,
					[WriteDescriptorSet::image_view_array(1, 0, image_view_writes.into_iter().flatten())],
					[]
				)?;

				Ok(Model {
					materials,
					material_variants,
					submeshes,
					vertex_subbuffers: vec![vbo_positions, vbo_texcoords, vbo_normals],
					index_buffer,
					path: path.to_path_buf(),
					textures_set,
					mat_tex_base_indices,
				})
			}
			_ => Err(format!("couldn't determine model file type of {}", path.display()).into()),
		}
	}

	pub fn new_model_instance(self: Arc<Self>, material_variant: Option<String>)
		-> Result<ModelInstance, GenericEngineError>
	{
		ModelInstance::new(self.clone(), material_variant)
	}

	pub fn path(&self) -> &Path
	{
		self.path.as_path()
	}

	/// Get the materials of this model.
	pub fn get_materials(&self) -> &Vec<Box<dyn Material>>
	{
		&self.materials
	}
	pub fn get_material_variants(&self) -> &Vec<String>
	{
		&self.material_variants
	}

	/// Draw this model. `transform` is the model/projection/view matrices multiplied for frustum culling.
	fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		pipeline_layout: Arc<PipelineLayout>,
		transform: &Mat4,
		transparency_pass: bool,
		shadow_pass: bool,
		material_variant: usize,
	) -> Result<(), GenericEngineError>
	{
		// TODO: Check each material's shader name so that we're only drawing them when the respective pipeline is bound.

		// Make sure that the material variant index given is not out of range.
		if let Some(first_submesh) = self.submeshes.first() {
			if material_variant >= first_submesh.mat_indices.len() {
				let err_str = format!(
					"Material variant index {} is out of range (there are {} material variants)",
					material_variant,
					first_submesh.mat_indices.len(),
				);
				return Err(err_str.into());
			}
		}

		// Determine which submeshes are visible.
		// "Visible" here means its transparency mode matches the current render pass type,
		// and the submesh passes frustum culling.
		let mut visible_submeshes = self
			.submeshes
			.iter()
			.filter(|submesh| {
				let material_index = submesh.mat_indices[material_variant];
				let mat = &self.materials[material_index];
				mat.has_transparency() == transparency_pass
			})
			.filter(|submesh| submesh.cull(transform))
			.peekable();

		// Don't even bother with binds if no submeshes are visible
		if visible_submeshes.peek().is_some() {
			let vertex_subbuffers = shadow_pass
				.then(|| vec![self.vertex_subbuffers[0].clone()])
				.unwrap_or_else(|| self.vertex_subbuffers.clone());

			cb.bind_vertex_buffers(0, vertex_subbuffers)?;
			self.index_buffer.bind(cb)?;

			if !shadow_pass {
				let set = self.textures_set.clone();
				cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout, 0, set)?;
			}

			for submesh in visible_submeshes {
				let mat_index = submesh.mat_indices[material_variant];
				let instance_index = self.mat_tex_base_indices[mat_index];

				submesh.draw(cb, instance_index)?;
			}
		}
		Ok(())
	}
}

#[derive(Deserialize)]
struct MaterialExtras
{
	#[serde(default)]
	external: bool,
}

fn load_gltf_material(mat: &gltf::Material, search_folder: &Path) -> Result<Box<dyn Material>, GenericEngineError>
{
	// Use an external material file if specified in the extras.
	// This can be specified in Blender by giving a material a custom property called "external"
	// with a boolean value of `true` (box is checked).
	let use_external = if let Some(extras) = mat.extras() {
		let parsed_extras: MaterialExtras = serde_json::from_str(extras.get())?;
		parsed_extras.external
	} else {
		false
	};

	if use_external {
		let material_name = mat
			.name()
			.ok_or("model wants an external material, but the glTF mesh material has no name")?;
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
	pub fn bind(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
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
	mat_indices: Vec<usize>,
	corner_min: Vec3,
	corner_max: Vec3,
}
impl SubMesh
{
	pub fn from_gltf_primitive(
		primitive: &gltf::Primitive,
		first_index: u32,
		vertex_offset: i32,
	) -> Result<Self, GenericEngineError>
	{
		let indices = primitive.indices().ok_or("no indices in glTF primitive")?;

		// Get the material index for each material variant. If this glTF document doesn't have
		// material variants, `mat_indices` will contain exactly one index, the material index
		// from the regular material.
		let mat_indices = if primitive.mappings().len() > 0 {
			primitive
				.mappings()
				.map(|mapping| mapping.material().index().unwrap_or(0))
				.collect()
		} else {
			vec![primitive.material().index().unwrap_or(0)]
		};

		Ok(SubMesh {
			first_index,
			index_count: indices.count().try_into()?,
			vertex_offset,
			mat_indices,
			corner_min: primitive.bounding_box().min.into(),
			corner_max: primitive.bounding_box().max.into(),
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

	pub fn draw(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, instance_index: u32)
		-> Result<(), GenericEngineError>
	{
		cb.draw_indexed(self.index_count, 1, self.first_index, self.vertex_offset, instance_index)?;
		Ok(())
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

pub struct ModelInstance
{
	model: Arc<Model>,
	material_variant: usize,

	pub model_matrix: Mat4,
}
impl ModelInstance
{
	fn new(model: Arc<Model>, material_variant: Option<String>) -> Result<Self, GenericEngineError>
	{
		let material_variant_index = if let Some(selected_variant_name) = material_variant {
			model
				.material_variants
				.iter()
				.position(|variant_name| variant_name == &selected_variant_name)
				.unwrap_or_else(|| {
					log::warn!(
						"Invalid material variant name '{}'! Falling back to default material.",
						&selected_variant_name
					);
					0
				})
		} else {
			0
		};

		Ok(Self {
			model,
			material_variant: material_variant_index,
			model_matrix: Default::default(),
		})
	}

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		pipeline_layout: Arc<PipelineLayout>,
		transparency_pass: bool,
		shadow_pass: bool,
		projview: &Mat4,
	) -> Result<(), GenericEngineError>
	{
		let materials = self.model.get_materials();

		// look for any materials with transparency enabled or disabled (depending on `transparency_pass`)
		let draw_this_mesh = materials
			.iter()
			.any(|mat| mat.has_transparency() == transparency_pass);
		if !draw_this_mesh {
			return Ok(()); // skip to the next mesh if none of the materials match this pass type
		}

		let projviewmodel = *projview * self.model_matrix;
		if shadow_pass {
			// TODO: also consider point lights, which require different matrices
			cb.push_constants(pipeline_layout.clone(), 0, projviewmodel)?;
		} else {
			let translation = self.model_matrix.w_axis.xyz();
			let push_data = MeshPushConstant {
				projviewmodel,
				model_x: self.model_matrix.x_axis.xyz().extend(translation.x),
				model_y: self.model_matrix.y_axis.xyz().extend(translation.y),
				model_z: self.model_matrix.z_axis.xyz().extend(translation.z),
			};
			cb.push_constants(pipeline_layout.clone(), 0, push_data)?;
		}

		self.model.draw(cb, pipeline_layout, &projviewmodel, transparency_pass, shadow_pass, self.material_variant)?;

		Ok(())
	}
}
#[derive(Clone, Copy, bytemuck::AnyBitPattern)]
#[repr(C)]
struct MeshPushConstant
{
	projviewmodel: Mat4,
	model_x: Vec4,
	model_y: Vec4,
	model_z: Vec4,
}
