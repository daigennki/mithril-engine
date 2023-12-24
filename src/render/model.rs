/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use glam::*;
use gltf::accessor::DataType;
use gltf::Semantic;
use serde::Deserialize;
use shipyard::EntityId;
use std::any::TypeId;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{PipelineBindPoint, PipelineLayout};

use crate::material::{pbr::PBR, ColorInput, Material};
use crate::render::RenderContext;
use crate::EngineError;

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
	pub fn new(render_ctx: &mut RenderContext, path: &Path) -> Result<Self, EngineError>
	{
		let parent_folder = path.parent().unwrap();

		log::info!("Loading glTF file '{}'...", path.display());

		let (doc, data_buffers, _) = gltf::import(&path).map_err(|e| EngineError::new("failed to load glTF file", e))?;

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
			material_variants
				.iter()
				.enumerate()
				.for_each(|(i, variant)| log::debug!("{i}: {}", &variant));
		}

		// Create submeshes from glTF "primitives".
		let primitives = doc
			.nodes()
			.filter_map(|node| node.mesh().map(|mesh| mesh.primitives()))
			.flatten();
		for prim in primitives {
			let positions_accessor = match prim.get(&Semantic::Positions) {
				Some(accessor) => accessor,
				None => continue,
			};
			let texcoords_accessor = match prim.get(&Semantic::TexCoords(0)) {
				Some(accessor) => accessor,
				None => continue,
			};
			let normals_accessor = match prim.get(&Semantic::Normals) {
				Some(accessor) => accessor,
				None => continue,
			};
			let indices_accessor = match prim.indices() {
				Some(accessor) => accessor,
				None => continue,
			};
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
				other => {
					let e = InvalidIndicesType { got: other };
					return Err(EngineError::new("failed to get index buffer data from glTF document", e));
				}
			};
			positions.extend_from_slice(get_buf_data(&positions_accessor, &data_buffers)?);
			texcoords.extend_from_slice(get_buf_data(&texcoords_accessor, &data_buffers)?);
			normals.extend_from_slice(get_buf_data(&normals_accessor, &data_buffers)?);

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
		let texcoords_offset: u64 = combined_data.len().try_into().unwrap();
		combined_data.append(&mut texcoords);
		let normals_offset: u64 = combined_data.len().try_into().unwrap();
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
			let mat_shader_inputs = mat.get_shader_inputs();
			let mut mat_image_views = Vec::with_capacity(mat_shader_inputs.len());
			for input in mat_shader_inputs {
				let tex = input.into_texture(parent_folder, render_ctx)?;
				mat_image_views.push(tex.view().clone());
			}
			last_tex_index_stride = mat_image_views.len().try_into().unwrap();
			image_view_writes.push(mat_image_views);

			let next_mat_tex_base_index = mat_tex_base_indices.last().copied().unwrap_or(0) + last_tex_index_stride;
			mat_tex_base_indices.push(next_mat_tex_base_index);
		}
		// There will be one extra unused element at the end of `mat_tex_base_indices`, so remove it,
		// then give the first material a base texture index of 0.
		mat_tex_base_indices.pop();
		mat_tex_base_indices.insert(0, 0);

		let texture_count = image_view_writes
			.iter()
			.flatten()
			.count()
			.try_into()
			.expect("too many image writes");
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
			[WriteDescriptorSet::image_view_array(
				1,
				0,
				image_view_writes.into_iter().flatten(),
			)],
			[],
		)
		.map_err(|e| EngineError::vulkan_error("failed to create material textures descriptor set", e))?;

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

	pub fn new_instance(self: Arc<Self>, material_variant: Option<String>) -> ModelInstance
	{
		ModelInstance::new(self, material_variant)
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
	/// Returns `Ok(true)` if any submeshes were drawn at all.
	fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		pipeline_name: Option<&str>,
		pipeline_layout: Arc<PipelineLayout>,
		projview: &Mat4,
		model_matrix: &Mat4,
		transparency_pass: bool,
		shadow_pass: bool,
		material_variant: usize,
		resources_bound: &mut bool,
	) -> bool
	{
		// TODO: Check each material's shader name so that we're only drawing them when the respective pipeline is bound.

		let projviewmodel = *projview * *model_matrix;

		// Determine which submeshes are visible.
		// "Visible" here means it uses the currently bound material pipeline,
		// its transparency mode matches the current render pass type,
		// and the submesh passes frustum culling.
		let mut visible_submeshes = self
			.submeshes
			.iter()
			.filter(|submesh| {
				let material_index = submesh.mat_indices[material_variant];
				let mat = &self.materials[material_index];

				// don't filter by material pipeline name if `None` was given
				let pipeline_matches = pipeline_name
					.map(|some_pl_name| mat.material_name() == some_pl_name)
					.unwrap_or(true);

				pipeline_matches && mat.has_transparency() == transparency_pass
			})
			.filter(|submesh| submesh.cull(&projviewmodel))
			.peekable();

		let any_visible = visible_submeshes.peek().is_some();

		// Don't even bother with binds if no submeshes are visible
		if any_visible {
			if shadow_pass {
				cb.push_constants(pipeline_layout.clone(), 0, projviewmodel).unwrap();
			} else {
				let translation = model_matrix.w_axis.xyz();
				let push_data = MeshPushConstant {
					projviewmodel,
					model_x: model_matrix.x_axis.xyz().extend(translation.x),
					model_y: model_matrix.y_axis.xyz().extend(translation.y),
					model_z: model_matrix.z_axis.xyz().extend(translation.z),
				};
				cb.push_constants(pipeline_layout.clone(), 0, push_data).unwrap();
			}

			if !*resources_bound {
				let vbo;
				if shadow_pass {
					vbo = vec![self.vertex_subbuffers[0].clone()];
				} else {
					let set = self.textures_set.clone();
					cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout, 0, set)
						.unwrap();

					vbo = self.vertex_subbuffers.clone();
				}

				cb.bind_vertex_buffers(0, vbo).unwrap();
				self.index_buffer.bind(cb);

				*resources_bound = true;
			}

			for submesh in visible_submeshes {
				let mat_index = submesh.mat_indices[material_variant];
				let instance_index = self.mat_tex_base_indices[mat_index];

				submesh.draw(cb, instance_index);
			}
		}

		any_visible
	}
}

#[derive(Deserialize)]
struct MaterialExtras
{
	#[serde(default)]
	external: bool,
}

fn load_gltf_material(mat: &gltf::Material, search_folder: &Path) -> Result<Box<dyn Material>, EngineError>
{
	// Use an external material file if specified in the extras.
	// This can be specified in Blender by giving a material a custom property called "external"
	// with a boolean value of `true` (box is checked).
	let use_external = if let Some(extras) = mat.extras() {
		let parse_result: Result<MaterialExtras, _> = serde_json::from_str(extras.get());
		match parse_result {
			Ok(parsed_extras) => parsed_extras.external,
			Err(e) => {
				log::error!("external materials unavailable because parsing glTF material extras failed: {e}");
				false
			}
		}
	} else {
		false
	};

	if use_external {
		let material_name = mat
			.name()
			.ok_or("model wants an external material, but the glTF material has no name")?;
		let mat_path = search_folder.join(material_name).with_extension("yaml");

		log::info!(
			"External material specified, loading material file '{}'...",
			mat_path.display()
		);

		let mat_file = File::open(&mat_path).map_err(|e| EngineError::new("failed to open material file", e))?;
		let deserialized_mat: Box<dyn Material> =
			serde_yaml::from_reader(mat_file).map_err(|e| EngineError::new("failed to parse material file", e))?;

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

#[derive(Debug)]
struct InvalidIndicesType
{
	got: DataType,
}
impl std::error::Error for InvalidIndicesType
{
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)>
	{
		None
	}
}
impl std::fmt::Display for InvalidIndicesType
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "expected u16 or u32 index buffer, got '{:?}'", self.got)
	}
}

enum IndexBufferVariant
{
	U16(Subbuffer<[u16]>),
	U32(Subbuffer<[u32]>),
}
impl IndexBufferVariant
{
	pub fn bind(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>)
	{
		match self {
			IndexBufferVariant::U16(buf) => cb.bind_index_buffer(buf.clone()),
			IndexBufferVariant::U32(buf) => cb.bind_index_buffer(buf.clone()),
		}
		.unwrap();
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
	pub fn from_gltf_primitive(primitive: &gltf::Primitive, first_index: u32, vertex_offset: i32) -> Result<Self, EngineError>
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
			index_count: indices.count().try_into().unwrap(),
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
	{
		cb.draw_indexed(self.index_count, 1, self.first_index, self.vertex_offset, instance_index)
			.unwrap();
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

#[derive(Debug)]
struct BufferTypeMismatch
{
	expected: DataType,
	got: &'static str,
}
impl std::error::Error for BufferTypeMismatch
{
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)>
	{
		None
	}
}
impl std::fmt::Display for BufferTypeMismatch
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "expected '{:?}', but given glTF buffer has `{:?}", self.expected, self.got)
	}
}

/// Get a slice of the part of the buffer that the accessor points to.
fn get_buf_data<'a, T: 'static>(accessor: &gltf::Accessor, buffers: &'a Vec<gltf::buffer::Data>)
	-> Result<&'a [T], EngineError>
{
	if TypeId::of::<T>() != data_type_to_id(accessor.data_type()) {
		let mismatch_error = BufferTypeMismatch {
			expected: accessor.data_type(),
			got: std::any::type_name::<T>(),
		};
		return Err(EngineError::new("failed to validate glTF buffer type", mismatch_error));
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
	fn new(model: Arc<Model>, material_variant: Option<String>) -> Self
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

		Self {
			model,
			material_variant: material_variant_index,
			model_matrix: Default::default(),
		}
	}

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		pipeline_name: Option<&str>,
		pipeline_layout: Arc<PipelineLayout>,
		transparency_pass: bool,
		shadow_pass: bool,
		projview: &Mat4,
		vbo_bound: &mut bool,
	) -> bool
	{
		self.model.draw(
			cb,
			pipeline_name,
			pipeline_layout,
			&projview,
			&self.model_matrix,
			transparency_pass,
			shadow_pass,
			self.material_variant,
			vbo_bound,
		)
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

pub struct ManagedModel
{
	model: Arc<Model>,
	users: HashMap<EntityId, ModelInstance>,
}
impl ManagedModel
{
	pub fn new(model: Arc<Model>) -> Self
	{
		Self {
			model,
			users: Default::default(),
		}
	}

	pub fn model(&self) -> &Arc<Model>
	{
		&self.model
	}

	pub fn new_user(&mut self, eid: EntityId, material_variant: Option<String>)
	{
		let model_instance = self.model.clone().new_instance(material_variant);
		self.users.insert(eid, model_instance);
	}

	pub fn set_model_matrix(&mut self, eid: EntityId, model_matrix: Mat4)
	{
		self.users.get_mut(&eid).unwrap().model_matrix = model_matrix;
	}

	pub fn cleanup(&mut self, eid: EntityId)
	{
		self.users.remove(&eid);
	}

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		pipeline_name: Option<&str>,
		pipeline_layout: Arc<PipelineLayout>,
		transparency_pass: bool,
		shadow_pass: bool,
		projview: &Mat4,
	) -> bool
	{
		let mut any_drawn = false;

		// TODO: We should set a separate `bool` for the material descriptor set to false if a model
		// instance has a custom material variant, so that the wrong resources don't get bound.
		let mut resources_bound = false;

		for user in self.users.values() {
			if user.draw(
				cb,
				pipeline_name,
				pipeline_layout.clone(),
				transparency_pass,
				shadow_pass,
				projview,
				&mut resources_bound,
			) {
				any_drawn = true;
			}
		}

		any_drawn
	}
}
