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
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderingInfo, CommandBufferUsage,
	PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo, SecondaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::{
	layout::{
		DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
	},
	DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{
	sampler::{Sampler, SamplerCreateInfo},
	view::ImageView,
};
use vulkano::pipeline::{
	graphics::viewport::Viewport,
	layout::{PipelineLayoutCreateInfo, PushConstantRange},
	GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

use crate::component::mesh::Mesh;
use crate::material::{pbr::PBR, ColorInput, Material, MaterialPipelines};
use crate::render::RenderContext;
use crate::EngineError;

/// 3D model
struct Model
{
	materials: Vec<Box<dyn Material>>,
	material_variants: Vec<String>,
	submeshes: Vec<SubMesh>,
	vertex_subbuffers: Vec<Subbuffer<[f32]>>,
	index_buffer: IndexBufferVariant,

	// Image views for all materials used by all material variants the glTF document comes with.
	// Uses variable descriptor count.
	textures_set: Arc<PersistentDescriptorSet>,

	// The texture base indices in the variable descriptor count of `textures_set`.
	// (to get the texture base index, use the material index as an index to this)
	mat_tex_base_indices: Vec<u32>,

	// All entities with a `Mesh` component that use this model.
	users: HashMap<EntityId, ModelInstance>,
}
impl Model
{
	fn new(
		render_ctx: &mut RenderContext,
		material_textures_set_layout: Arc<DescriptorSetLayout>,
		path: &Path,
	) -> crate::Result<Self>
	{
		let parent_folder = path.parent().unwrap();

		log::info!("Loading glTF file '{}'...", path.display());

		let (doc, data_buffers, _) = gltf::import(&path).map_err(|e| EngineError::new("failed to load glTF file", e))?;

		let (vertex_subbuffers, index_buffer, submeshes) = load_gltf_meshes(render_ctx, &doc, &data_buffers)?;

		// Collect materials
		let materials: Vec<_> = doc
			.materials()
			.map(|mat| load_gltf_material(&mat, parent_folder))
			.collect::<Result<_, _>>()?;

		// Check what shaders are being used across all of the materials in this model.
		let mut shader_names = BTreeSet::new();
		for mat in &materials {
			if !shader_names.contains(mat.material_name()) {
				shader_names.insert(mat.material_name());
			}
		}
		log::debug!("Model has {} submeshes. It uses these shaders:", submeshes.len());
		shader_names.iter().for_each(|shader_name| log::debug!("- {shader_name}"));

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
			for (i, variant) in material_variants.iter().enumerate() {
				log::debug!("{i}: {variant}");
			}
		}

		if material_variants.is_empty() && submeshes.len() > materials.len() {
			log::warn!(
				r"There are more meshes than materials in the model, even though there are no material variants!
This model may be inefficient to draw, so consider joining the meshes."
			);
		}

		let (textures_set, mat_tex_base_indices) =
			descriptor_set_from_materials(render_ctx, material_textures_set_layout, parent_folder, &materials)?;

		Ok(Model {
			materials,
			material_variants,
			submeshes,
			vertex_subbuffers,
			index_buffer,
			textures_set,
			mat_tex_base_indices,
			users: Default::default(),
		})
	}

	fn new_user(&mut self, eid: EntityId, material_variant: Option<String>)
	{
		self.users.insert(eid, ModelInstance::new(&self, material_variant));
	}
	fn set_model_matrix(&mut self, eid: EntityId, model_matrix: Mat4)
	{
		self.users.get_mut(&eid).unwrap().model_matrix = model_matrix;
	}
	fn cleanup(&mut self, eid: EntityId)
	{
		self.users.remove(&eid);
	}

	/// Draw this model for all visible users. Returns `Ok(true)` if any submeshes were drawn at all.
	fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		pipeline_name: Option<&str>,
		pipeline_layout: Arc<PipelineLayout>,
		transparency_pass: bool,
		shadow_pass: bool,
		projview: &Mat4,
	) -> crate::Result<bool>
	{
		let mut any_drawn = false;

		// TODO: We should set a separate `bool` for the material descriptor set to false if a model
		// instance has a custom material variant, so that the wrong resources don't get bound.
		let mut resources_bound = false;

		for user in self.users.values() {
			let model_matrix = &user.model_matrix;
			let material_variant = user.material_variant;

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

			// Don't even bother with binds if no submeshes are visible in this model instance
			if visible_submeshes.peek().is_some() {
				if shadow_pass {
					cb.push_constants(pipeline_layout.clone(), 0, projviewmodel)?;
				} else {
					let translation = model_matrix.w_axis.xyz();
					let push_data = MeshPushConstant {
						projviewmodel,
						model_x: model_matrix.x_axis.xyz().extend(translation.x),
						model_y: model_matrix.y_axis.xyz().extend(translation.y),
						model_z: model_matrix.z_axis.xyz().extend(translation.z),
					};
					cb.push_constants(pipeline_layout.clone(), 0, push_data)?;
				}

				if !resources_bound {
					let vbo;
					if shadow_pass {
						vbo = vec![self.vertex_subbuffers[0].clone()];
					} else {
						let set = self.textures_set.clone();
						cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 0, set)?;

						vbo = self.vertex_subbuffers.clone();
					}

					cb.bind_vertex_buffers(0, vbo)?;
					self.index_buffer.bind(cb)?;

					resources_bound = true;
				}

				for submesh in visible_submeshes {
					let mat_index = submesh.mat_indices[material_variant];
					let instance_index = self.mat_tex_base_indices[mat_index];

					submesh.draw(cb, instance_index)?;
				}

				any_drawn = true;
			}
		}

		Ok(any_drawn)
	}
}

#[derive(Deserialize)]
struct MaterialExtras
{
	#[serde(default)]
	external: bool,
}

fn load_gltf_material(mat: &gltf::Material, search_folder: &Path) -> crate::Result<Box<dyn Material>>
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

fn load_gltf_meshes(
	render_ctx: &mut RenderContext,
	doc: &gltf::Document,
	data_buffers: &[gltf::buffer::Data],
) -> crate::Result<(Vec<Subbuffer<[f32]>>, IndexBufferVariant, Vec<SubMesh>)>
{
	// Collect all of the vertex data into buffers shared by all submeshes to reduce the number of binds.
	let mut submeshes = Vec::new();
	let mut positions = Vec::new();
	let mut texcoords = Vec::new();
	let mut normals = Vec::new();
	let mut indices_u16 = Vec::new();
	let mut indices_u32 = Vec::new();

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
			Some(accessor) => match accessor.data_type() {
				DataType::U16 | DataType::U32 => accessor,
				other => {
					log::warn!("glTF primitive has invalid index buffer data type '{other:?}', skipping primitive...");
					continue;
				}
			},
			None => continue,
		};

		let first_index = indices_u32.len().max(indices_u16.len()).try_into().unwrap();
		let vertex_offset = (positions.len() / 3).try_into().unwrap();
		let submesh = SubMesh::from_gltf_primitive(&prim, first_index, vertex_offset)?;
		submeshes.push(submesh);

		match indices_accessor.data_type() {
			DataType::U16 => {
				let slice_u16: &[u16] = get_buf_data(&indices_accessor, data_buffers)?;
				if indices_u32.len() > 0 {
					// convert the indices to u32 if mixed with u32 indices
					indices_u32.extend(slice_u16.iter().copied().map(|index| index as u32));
				} else {
					indices_u16.extend_from_slice(slice_u16);
				}
			}
			DataType::U32 => {
				if indices_u16.len() > 0 {
					// convert existing indices to u32 if they're u16
					indices_u32 = indices_u16.drain(..).map(|index| index as u32).collect();
					indices_u16 = Vec::new(); // free some memory
				}
				indices_u32.extend_from_slice(get_buf_data(&indices_accessor, data_buffers)?);
			}
			_ => unreachable!(),
		};

		positions.extend_from_slice(get_buf_data(&positions_accessor, data_buffers)?);
		texcoords.extend_from_slice(get_buf_data(&texcoords_accessor, data_buffers)?);
		normals.extend_from_slice(get_buf_data(&normals_accessor, data_buffers)?);
	}

	// Combine the vertex data into a single buffer,
	// then split it into subbuffers for different types of vertex data.
	let mut combined_data = Vec::with_capacity(positions.len() + texcoords.len() + normals.len());
	combined_data.append(&mut positions);
	let texcoords_offset: u64 = combined_data.len().try_into().unwrap();
	combined_data.append(&mut texcoords);
	let normals_offset: u64 = combined_data.len().try_into().unwrap();
	combined_data.append(&mut normals);

	let vertex_buffer = render_ctx.new_buffer(combined_data.as_slice(), BufferUsage::VERTEX_BUFFER)?;
	let vbo_positions = vertex_buffer.clone().slice(..texcoords_offset);
	let vbo_texcoords = vertex_buffer.clone().slice(texcoords_offset..normals_offset);
	let vbo_normals = vertex_buffer.clone().slice(normals_offset..);
	let vertex_subbuffers = vec![vbo_positions, vbo_texcoords, vbo_normals];

	let index_buffer = if indices_u32.len() > 0 {
		IndexBufferVariant::U32(render_ctx.new_buffer(&indices_u32, BufferUsage::INDEX_BUFFER)?)
	} else {
		IndexBufferVariant::U16(render_ctx.new_buffer(&indices_u16, BufferUsage::INDEX_BUFFER)?)
	};

	Ok((vertex_subbuffers, index_buffer, submeshes))
}

fn descriptor_set_from_materials(
	render_ctx: &mut RenderContext,
	material_textures_set_layout: Arc<DescriptorSetLayout>,
	parent_folder: &Path,
	materials: &[Box<dyn Material>],
) -> crate::Result<(Arc<PersistentDescriptorSet>, Vec<u32>)>
{
	// Get the image views for each material, and calculate the base index in the variable descriptor count.
	let mut image_view_writes = Vec::with_capacity(materials.len());
	let mut mat_tex_base_indices = Vec::with_capacity(materials.len());
	let mut last_tex_index_stride = 0;
	for mat in materials {
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
		.map(|write| write.len())
		.sum::<usize>()
		.try_into()
		.expect("too many image writes");
	log::debug!("texture count (variable descriptor count): {}", texture_count);

	// Make sure that the shader doesn't overrun the variable count descriptor.
	// Some very weird things (like crashing the entire computer) might happen if we don't check this!
	let last_mat_tex_base_index = mat_tex_base_indices.last().unwrap();
	assert!(last_mat_tex_base_index + last_tex_index_stride <= texture_count);

	// Make a single write out of the image views of all of the materials, and create a single descriptor set.
	let set_alloc = &render_ctx.descriptor_set_allocator;
	let set_layout = material_textures_set_layout;
	let write = WriteDescriptorSet::image_view_array(1, 0, image_view_writes.into_iter().flatten());
	let textures_set = PersistentDescriptorSet::new_variable(set_alloc, set_layout, texture_count, [write], [])?;

	Ok((textures_set, mat_tex_base_indices))
}

enum IndexBufferVariant
{
	U16(Subbuffer<[u16]>),
	U32(Subbuffer<[u32]>),
}
impl IndexBufferVariant
{
	pub fn bind(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>) -> crate::Result<()>
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
	fn from_gltf_primitive(primitive: &gltf::Primitive, first_index: u32, vertex_offset: i32) -> crate::Result<Self>
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
	fn cull(&self, projviewmodel: &Mat4) -> bool
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

	fn draw(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, instance_index: u32) -> crate::Result<()>
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
fn get_buf_data<'a, T: 'static>(accessor: &gltf::Accessor, buffers: &'a [gltf::buffer::Data]) -> crate::Result<&'a [T]>
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

struct ModelInstance
{
	material_variant: usize,
	model_matrix: Mat4,
}
impl ModelInstance
{
	fn new(model: &Model, material_variant: Option<String>) -> Self
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
			material_variant: material_variant_index,
			model_matrix: Default::default(),
		}
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

/// A single manager that manages the GPU resources for all `Mesh` components.
#[derive(shipyard::Unique)]
pub struct MeshManager
{
	material_textures_set_layout: Arc<DescriptorSetLayout>,

	pipeline_layout: Arc<PipelineLayout>,
	pipeline_layout_oit: Arc<PipelineLayout>,

	material_pipelines: BTreeMap<&'static str, MaterialPipelines>,

	// Loaded 3D models, with the key being the path relative to the current working directory.
	// Each model will also contain the model instance for each entity.
	models: HashMap<PathBuf, Model>,

	// A mapping between entity IDs and the model it uses.
	resources: HashMap<EntityId, PathBuf>,

	cb_3d: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,
}
impl MeshManager
{
	pub fn new(render_ctx: &mut RenderContext) -> crate::Result<Self>
	{
		let vk_dev = render_ctx.memory_allocator.device().clone();

		/* Common material texture descriptor set layout */
		let mat_tex_sampler_info = SamplerCreateInfo {
			anisotropy: Some(16.0),
			..SamplerCreateInfo::simple_repeat_linear()
		};
		let mat_tex_sampler = Sampler::new(vk_dev.clone(), mat_tex_sampler_info)?;
		let mat_tex_bindings = [
			DescriptorSetLayoutBinding {
				// binding 0: sampler0
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![mat_tex_sampler],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
			},
			DescriptorSetLayoutBinding {
				// binding 1: textures
				binding_flags: DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
				descriptor_count: 32,
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
			},
		];
		let mat_tex_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: (0..).zip(mat_tex_bindings).collect(),
			..Default::default()
		};
		let material_textures_set_layout = DescriptorSetLayout::new(vk_dev.clone(), mat_tex_set_layout_info)?;

		let light_set_layout = render_ctx.light_set_layout.clone();

		render_ctx.load_transparency(material_textures_set_layout.clone())?;
		let transparency_input_layout = render_ctx
			.transparency_renderer
			.as_ref()
			.unwrap()
			.get_stage3_inputs()
			.layout()
			.clone();

		let push_constant_size = std::mem::size_of::<Mat4>() + std::mem::size_of::<Vec4>() * 3 + std::mem::size_of::<UVec2>();
		let push_constant_range = PushConstantRange {
			stages: ShaderStages::VERTEX,
			offset: 0,
			size: push_constant_size.try_into().unwrap(),
		};
		let layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![material_textures_set_layout.clone(), light_set_layout.clone()],
			push_constant_ranges: vec![push_constant_range],
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(vk_dev.clone(), layout_info)?;

		let layout_info_oit = PipelineLayoutCreateInfo {
			set_layouts: vec![
				material_textures_set_layout.clone(),
				light_set_layout,
				transparency_input_layout,
			],
			push_constant_ranges: vec![push_constant_range],
			..Default::default()
		};
		let pipeline_layout_oit = PipelineLayout::new(vk_dev.clone(), layout_info_oit)?;

		Ok(Self {
			material_textures_set_layout,
			pipeline_layout,
			pipeline_layout_oit,
			material_pipelines: Default::default(),
			models: Default::default(),
			resources: Default::default(),
			cb_3d: Default::default(),
		})
	}

	/// Load the model for the given `Mesh`.
	pub fn load(&mut self, render_ctx: &mut RenderContext, eid: EntityId, component: &Mesh) -> crate::Result<()>
	{
		// Get a 3D model from `path`, relative to the current working directory.
		// This attempts loading if it hasn't been loaded into memory yet.
		let loaded_model = match self.models.get_mut(&component.model_path) {
			Some(m) => m,
			None => {
				let new_model = Model::new(render_ctx, self.material_textures_set_layout.clone(), &component.model_path)?;
				self.models.insert(component.model_path.clone(), new_model);
				self.models.get_mut(&component.model_path).unwrap()
			}
		};

		// Go through all the materials, and load the pipelines they need if they aren't already loaded.
		for mat in &loaded_model.materials {
			let mat_name = mat.material_name();
			if !self.material_pipelines.contains_key(mat_name) {
				let pipeline_config = mat.load_shaders(self.pipeline_layout.device().clone())?;
				let pipeline_data = pipeline_config.into_pipelines(
					render_ctx.depth_stencil_format(),
					self.pipeline_layout.clone(),
					self.pipeline_layout_oit.clone(),
				)?;

				self.material_pipelines.insert(mat_name, pipeline_data);
			}
		}

		loaded_model.new_user(eid, component.material_variant.clone());
		self.resources.insert(eid, component.model_path.clone());

		Ok(())
	}

	pub fn set_model_matrix(&mut self, eid: EntityId, model_matrix: Mat4)
	{
		let path = self.resources.get(&eid).unwrap().as_path();
		self.models.get_mut(path).unwrap().set_model_matrix(eid, model_matrix);
	}

	/// Free resources for the given entity ID. Only call this when the `Mesh` component was actually removed!
	pub fn cleanup_removed(&mut self, eid: EntityId)
	{
		let path = self.resources.get(&eid).unwrap().as_path();
		self.models.get_mut(path).unwrap().cleanup(eid);
		self.resources.remove(&eid);
	}

	pub fn draw(
		&self,
		render_ctx: &RenderContext,
		projview: Mat4,
		pass_type: PassType,
		common_sets: &[Arc<PersistentDescriptorSet>],
	) -> crate::Result<Option<Arc<SecondaryAutoCommandBuffer>>>
	{
		let (depth_format, viewport_extent, shadow_pass) = match &pass_type {
			PassType::Shadow {
				format, viewport_extent, ..
			} => (*format, *viewport_extent, true),
			_ => (render_ctx.depth_stencil_format(), render_ctx.swapchain_dimensions(), false),
		};

		let rendering_inheritance = CommandBufferInheritanceRenderingInfo {
			color_attachment_formats: pass_type.render_color_formats(),
			depth_attachment_format: Some(depth_format),
			stencil_attachment_format: pass_type.needs_stencil_buffer().then_some(depth_format),
			..Default::default()
		};
		let mut cb = AutoCommandBufferBuilder::secondary(
			&render_ctx.command_buffer_allocator,
			render_ctx.graphics_queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
			CommandBufferInheritanceInfo {
				render_pass: Some(rendering_inheritance.into()),
				..Default::default()
			},
		)?;

		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [viewport_extent[0] as f32, viewport_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};
		cb.set_viewport(0, [viewport].as_slice().into())?;

		let pipeline_override = pass_type.pipeline();
		let transparency_pass = pass_type.transparency_pass();

		let mut any_drawn = false;
		for (pipeline_name, mat_pl) in &self.material_pipelines {
			let pipeline = if let Some(pl) = pipeline_override {
				pl.clone()
			} else if transparency_pass {
				if let Some(pl) = mat_pl.oit_pipeline.clone() {
					pl
				} else {
					continue;
				}
			} else {
				mat_pl.opaque_pipeline.clone()
			};

			cb.bind_pipeline_graphics(pipeline.clone())?;
			let pipeline_layout = pipeline.layout().clone();

			if pass_type.needs_viewport_extent_push_constant() {
				cb.push_constants(pipeline_layout.clone(), 112, viewport_extent)?;
			}

			if common_sets.len() > 0 {
				let sets = Vec::from(common_sets);
				cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 1, sets)?;
			}

			// don't filter by material pipeline name if there is a pipeline override
			let pipeline_name_option = pipeline_override.is_none().then_some(pipeline_name);

			for model in self.models.values() {
				if model.draw(
					&mut cb,
					pipeline_name_option.copied(),
					pipeline_layout.clone(),
					transparency_pass,
					shadow_pass,
					&projview,
				)? {
					any_drawn = true;
				}
			}

			if pipeline_override.is_some() {
				break;
			}
		}

		// Don't bother building the command buffer if this is the OIT pass and no models were drawn.
		let cb_return = if any_drawn || !transparency_pass || shadow_pass {
			Some(cb.build()?)
		} else {
			None
		};
		Ok(cb_return)
	}

	pub fn add_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		*self.cb_3d.lock().unwrap() = Some(cb);
	}

	pub fn execute_rendering(
		&mut self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		color_image: Arc<ImageView>,
		depth_image: Arc<ImageView>,
	) -> crate::Result<()>
	{
		let main_render_info = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(color_image)
			})],
			depth_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store, // order-independent transparency needs this to be `Store`
				clear_value: Some(ClearValue::Depth(1.0)),
				..RenderingAttachmentInfo::image_view(depth_image)
			}),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		};
		let secondary_cb = self.cb_3d.lock().unwrap().take().unwrap();
		cb_builder
			.begin_rendering(main_render_info)?
			.execute_commands(secondary_cb)?
			.end_rendering()?;

		Ok(())
	}
}

pub enum PassType
{
	Shadow
	{
		pipeline: Arc<GraphicsPipeline>,
		format: Format,
		viewport_extent: [u32; 2],
	},
	Opaque,
	TransparencyMoments(Arc<GraphicsPipeline>),
	Transparency,
}
impl PassType
{
	fn render_color_formats(&self) -> Vec<Option<Format>>
	{
		let formats: &'static [Format] = match self {
			PassType::Shadow { .. } => &[],
			PassType::Opaque => &[Format::R16G16B16A16_SFLOAT],
			PassType::TransparencyMoments(_) => {
				&[Format::R32G32B32A32_SFLOAT, Format::R32_SFLOAT /*, Format::R32_SFLOAT*/]
			}
			PassType::Transparency => &[Format::R16G16B16A16_SFLOAT, Format::R8_UNORM],
		};
		formats.iter().copied().map(|f| Some(f)).collect()
	}
	fn pipeline(&self) -> Option<&Arc<GraphicsPipeline>>
	{
		match self {
			PassType::Shadow { pipeline, .. } | PassType::TransparencyMoments(pipeline) => Some(pipeline),
			_ => None,
		}
	}
	fn transparency_pass(&self) -> bool
	{
		match self {
			PassType::TransparencyMoments(_) | PassType::Transparency => true,
			_ => false,
		}
	}
	fn needs_stencil_buffer(&self) -> bool
	{
		match self {
			PassType::Transparency => true,
			_ => false,
		}
	}
	fn needs_viewport_extent_push_constant(&self) -> bool
	{
		match self {
			PassType::Opaque | PassType::Transparency => true,
			_ => false,
		}
	}
}
