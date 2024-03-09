/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use glam::*;
use gltf::{mesh::util::ReadIndices, Semantic};
use serde::Deserialize;
use shipyard::{EntityId, UniqueView};
use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use vulkano::buffer::{BufferUsage, IndexBuffer, Subbuffer};
use vulkano::command_buffer::*;
use vulkano::descriptor_set::{allocator::*, layout::*, *};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{sampler::*, view::ImageView, SampleCount};
use vulkano::pipeline::graphics::{vertex_input::*, viewport::Viewport};
use vulkano::pipeline::{layout::*, *};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;
use vulkano::DeviceSize;

use super::lighting::LightManager;
use super::RenderContext;
use crate::component::{camera::CameraManager, mesh::Mesh};
use crate::material::{pbr::PBR, BlendMode, Material, MaterialPipelineConfig, MaterialPipelines};
use crate::EngineError;

/// Vertex attributes and bindings describing the vertex buffers bound by a model. The indices in
/// each slice are used as the binding and attribute indices respectively as-is.
pub const VERTEX_BINDINGS: [VertexInputBindingDescription; 3] = [
	VertexInputBindingDescription {
		stride: 12,
		input_rate: VertexInputRate::Vertex,
	},
	VertexInputBindingDescription {
		stride: 8,
		input_rate: VertexInputRate::Vertex,
	},
	VertexInputBindingDescription {
		stride: 12,
		input_rate: VertexInputRate::Vertex,
	},
];
pub const VERTEX_ATTRIBUTES: [VertexInputAttributeDescription; 3] = [
	VertexInputAttributeDescription {
		binding: 0,
		format: Format::R32G32B32_SFLOAT,
		offset: 0,
	},
	VertexInputAttributeDescription {
		binding: 1,
		format: Format::R32G32_SFLOAT,
		offset: 0,
	},
	VertexInputAttributeDescription {
		binding: 2,
		format: Format::R32G32B32_SFLOAT,
		offset: 0,
	},
];

/// 3D model
struct Model
{
	materials: Vec<Box<dyn Material>>,
	material_variants: Vec<String>,
	submeshes: Vec<SubMesh>,
	vertex_subbuffers: Vec<Subbuffer<[f32]>>,
	index_buffer: IndexBuffer,

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
		descriptor_set_allocator: &StandardDescriptorSetAllocator,
		material_textures_set_layout: Arc<DescriptorSetLayout>,
		path: &Path,
	) -> crate::Result<Self>
	{
		log::info!("Loading glTF file '{}'...", path.display());

		let (doc, data_buffers, _) = gltf::import(path).map_err(|e| EngineError::new("failed to load glTF file", e))?;

		let (vertex_subbuffers, index_buffer, submeshes) = load_gltf_meshes(render_ctx, &doc, &data_buffers)?;

		let parent_folder = path.parent().unwrap();
		let materials: Vec<_> = doc.materials().map(|mat| load_gltf_material(&mat, parent_folder)).collect();
		let (textures_set, mat_tex_base_indices) = descriptor_set_from_materials(
			render_ctx,
			descriptor_set_allocator,
			material_textures_set_layout,
			parent_folder,
			&materials,
		)?;

		let material_variants: Vec<_> = doc
			.variants()
			.into_iter()
			.flatten()
			.map(|variant| variant.name().to_string())
			.collect();
		if material_variants.is_empty() {
			log::debug!("no material variants in model");
			if submeshes.len() > materials.len() {
				log::warn!(
					"There are more meshes than materials in the model, even though there are no material variants! \
					This model may be inefficient to draw, so consider joining the meshes."
				);
			}
		} else {
			log::debug!("material variants in model: {:?}", &material_variants);
		}

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
		self.users.insert(eid, ModelInstance::new(self, material_variant));
	}
	fn set_affine(&mut self, eid: EntityId, affine: DAffine3)
	{
		self.users.get_mut(&eid).unwrap().affine = affine;
	}
	fn cleanup(&mut self, eid: EntityId)
	{
		self.users.remove(&eid);
	}

	/// Draw this model's submeshes for all visible users. Returns `true` if any submeshes were
	/// drawn at all.
	fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		pipeline_name: Option<TypeId>,
		pipeline_layout: Arc<PipelineLayout>,
		transparency_pass: bool,
		shadow_pass: bool,
		projview: &DMat4,
	) -> bool
	{
		// The blend mode to render for during this pass.
		let blend_mode_this_pass = match transparency_pass {
			true => BlendMode::AlphaBlend,
			false => BlendMode::Opaque,
		};

		// TODO: We should separately handle users with custom material variants after ones without
		// custom material variants, so that the wrong descriptor set doesn't get bound.
		let mut any_drawn = false;

		for user in self.users.values() {
			let projviewmodel = *projview * DMat4::from(user.affine);

			// Convert the matrix values from f64 to f32 only after multiplying the matrices. This
			// prevents precision loss, especially with large values in the affine transformation.
			let pvm_f32 = projviewmodel.as_mat4();

			// Filter visible submeshes. "Visible" here means it uses the currently bound material
			// pipeline, its transparency mode matches the current render pass type, and the submesh
			// passes frustum culling.
			let mut visible_submeshes = self
				.submeshes
				.iter()
				.filter(|submesh| {
					let material_index = submesh.mat_indices[user.material_variant];
					let mat = &self.materials[material_index];

					// don't filter by material pipeline name if `None` was given
					let pipeline_matches = pipeline_name
						.map(|some_pl_name| mat.as_ref().type_id() == some_pl_name)
						.unwrap_or(true);

					pipeline_matches && mat.blend_mode() == blend_mode_this_pass
				})
				.filter(|submesh| submesh.cull(&pvm_f32))
				.peekable();

			// Don't even bother with binds if no submeshes are visible in this model instance.
			if visible_submeshes.peek().is_some() {
				if shadow_pass {
					cb.push_constants(pipeline_layout.clone(), 0, pvm_f32).unwrap();
				} else {
					let affine_mat_f32 = user.affine.matrix3.as_mat3();
					let translation = user.affine.translation.as_vec3();
					let push_data = MeshPushConstant {
						projviewmodel: pvm_f32,
						model_x: affine_mat_f32.x_axis.extend(translation.x),
						model_y: affine_mat_f32.y_axis.extend(translation.y),
						model_z: affine_mat_f32.z_axis.extend(translation.z),
					};
					cb.push_constants(pipeline_layout.clone(), 0, push_data).unwrap();
				}

				if !any_drawn {
					let vbo = if shadow_pass {
						vec![self.vertex_subbuffers[0].clone()]
					} else {
						let set = self.textures_set.clone();
						cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 0, set)
							.unwrap();
						self.vertex_subbuffers.clone()
					};

					cb.bind_vertex_buffers(0, vbo).unwrap();
					cb.bind_index_buffer(self.index_buffer.clone()).unwrap();
				}

				for submesh in visible_submeshes {
					let mat_index = submesh.mat_indices[user.material_variant];
					let instance_index = self.mat_tex_base_indices[mat_index];
					submesh.draw(cb, instance_index);
				}

				any_drawn = true;
			}
		}

		any_drawn
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
	fn from_gltf_primitive(primitive: &gltf::Primitive, first_index: u32, vertex_offset: i32) -> Self
	{
		// Get the material index for each material variant. If the glTF document has no material
		// variants, `mat_indices` will contain only the material index of the regular material.
		let mat_indices = if primitive.mappings().len() > 0 {
			primitive
				.mappings()
				.map(|mapping| mapping.material().index().unwrap_or(0))
				.collect()
		} else {
			vec![primitive.material().index().unwrap_or(0)]
		};

		SubMesh {
			first_index,
			index_count: primitive.indices().unwrap().count().try_into().unwrap(),
			vertex_offset,
			mat_indices,
			corner_min: primitive.bounding_box().min.into(),
			corner_max: primitive.bounding_box().max.into(),
		}
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

	fn draw(&self, cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, instance_index: u32)
	{
		cb.draw_indexed(self.index_count, 1, self.first_index, self.vertex_offset, instance_index)
			.unwrap();
	}
}

// (vertex buffers, index buffer, submeshes)
type LoadedMeshData = (Vec<Subbuffer<[f32]>>, IndexBuffer, Vec<SubMesh>);
fn load_gltf_meshes(
	render_ctx: &mut RenderContext,
	doc: &gltf::Document,
	data_buffers: &[gltf::buffer::Data],
) -> crate::Result<LoadedMeshData>
{
	// Collect all of the vertex data into buffers shared by all submeshes to reduce the number of binds.
	let mut submeshes = Vec::new();
	let mut positions = Vec::new();
	let mut texcoords = Vec::new();
	let mut normals = Vec::new();
	let mut indices_u16 = Vec::new();
	let mut indices_u32 = Vec::new();

	// Skip the primitive if it doesn't have all the semantics we need.
	const REQUIRED_SEMANTICS: [Semantic; 3] = [Semantic::Positions, Semantic::Normals, Semantic::TexCoords(0)];

	// Create submeshes from glTF "primitives".
	let primitives = doc
		.meshes()
		.flat_map(|mesh| mesh.primitives())
		.filter(|prim| REQUIRED_SEMANTICS.iter().all(|s| prim.get(s).is_some()) && prim.indices().is_some());
	for prim in primitives {
		let first_index = indices_u32.len().max(indices_u16.len()).try_into().unwrap();
		let vertex_offset = (positions.len() / 3).try_into().unwrap();
		submeshes.push(SubMesh::from_gltf_primitive(&prim, first_index, vertex_offset));

		let reader = prim.reader(|buf| data_buffers.get(buf.index()).map(|data| data.0.as_slice()));
		positions.extend(reader.read_positions().unwrap().flatten());
		texcoords.extend(reader.read_tex_coords(0).unwrap().into_f32().flatten());
		normals.extend(reader.read_normals().unwrap().flatten());
		match reader.read_indices().unwrap() {
			ReadIndices::U8(read_u8) => {
				if !indices_u32.is_empty() {
					indices_u32.extend(read_u8.map(|index| index as u32));
				} else {
					indices_u16.extend(read_u8.map(|index| index as u16));
				}
			}
			ReadIndices::U16(read_u16) => {
				if !indices_u32.is_empty() {
					indices_u32.extend(read_u16.map(|index| index as u32));
				} else {
					indices_u16.extend(read_u16);
				}
			}
			ReadIndices::U32(iter_u32) => {
				if !indices_u16.is_empty() {
					// convert existing indices to u32 if they're u16
					indices_u32 = indices_u16.drain(..).map(|index| index as u32).collect();
					indices_u16.shrink_to_fit(); // free some memory
				}
				indices_u32.extend(iter_u32);
			}
		}
	}

	// Combine the vertex data into a single buffer, then split it into subbuffers for different
	// types of vertex data.
	let texcoords_offset: DeviceSize = positions.len().try_into().unwrap();
	let normals_offset = texcoords_offset + DeviceSize::try_from(texcoords.len()).unwrap();
	let combined_data = [positions, texcoords, normals].concat();

	let vertex_buffer = render_ctx.new_buffer(&combined_data, BufferUsage::VERTEX_BUFFER)?;
	let vbo_positions = vertex_buffer.clone().slice(..texcoords_offset);
	let vbo_texcoords = vertex_buffer.clone().slice(texcoords_offset..normals_offset);
	let vbo_normals = vertex_buffer.clone().slice(normals_offset..);
	let vertex_subbuffers = vec![vbo_positions, vbo_texcoords, vbo_normals];

	let index_buffer = if !indices_u32.is_empty() {
		IndexBuffer::U32(render_ctx.new_buffer(&indices_u32, BufferUsage::INDEX_BUFFER)?)
	} else {
		IndexBuffer::U16(render_ctx.new_buffer(&indices_u16, BufferUsage::INDEX_BUFFER)?)
	};

	Ok((vertex_subbuffers, index_buffer, submeshes))
}

#[derive(Deserialize)]
struct MaterialExtras
{
	#[serde(default)]
	external: bool,
}
fn load_gltf_material(mat: &gltf::Material, search_folder: &Path) -> Box<dyn Material>
{
	// Use an external material file if specified in the extras. This can be specified in Blender by
	// giving a material a custom property called "external" with a boolean value of `true` (box is
	// checked). This will fall back to the embedded glTF material if this fails.
	if let Some(extras) = mat.extras() {
		match serde_json::from_str(extras.get()) {
			Ok(MaterialExtras { external: true }) => match mat.name() {
				Some(name) => {
					let mat_path = search_folder.join(name).with_extension("yaml");
					log::info!("Loading external material file '{}'...", mat_path.display());
					match File::open(&mat_path) {
						Ok(file) => match serde_yaml::from_reader(file) {
							Ok(mat) => return mat,
							Err(e) => log::error!("failed to parse material file: {e}"),
						},
						Err(e) => log::error!("failed to open material file: {e}"),
					}
				}
				None => log::error!("a model wants an external material, but the glTF material has no name"),
			},
			Err(e) => log::error!("external material unavailable because parsing glTF material extras failed: {e}"),
			_ => (),
		}
	}

	Box::new(PBR::from(mat))
}
fn descriptor_set_from_materials(
	render_ctx: &mut RenderContext,
	set_alloc: &StandardDescriptorSetAllocator,
	set_layout: Arc<DescriptorSetLayout>,
	parent_folder: &Path,
	materials: &[Box<dyn Material>],
) -> crate::Result<(Arc<PersistentDescriptorSet>, Vec<u32>)>
{
	// Get the image views for each material, and calculate the base index in the variable descriptor count.
	let mut image_views = Vec::with_capacity(materials.len());
	let mut mat_tex_base_indices = Vec::with_capacity(materials.len());
	let mut last_tex_index_stride = 0;
	for mat in materials {
		let mat_image_views: Vec<_> = mat
			.get_shader_inputs()
			.into_iter()
			.map(|input| input.into_texture(parent_folder, render_ctx))
			.collect::<Result<_, _>>()?;

		let mat_tex_base_index = mat_tex_base_indices.last().copied().unwrap_or(0) + last_tex_index_stride;
		last_tex_index_stride = mat_image_views.len().try_into().unwrap();

		image_views.push(mat_image_views);
		mat_tex_base_indices.push(mat_tex_base_index);
	}

	// Make a single write out of the image views of all of the materials, and create a single descriptor set.
	let write = WriteDescriptorSet::image_view_array(1, 0, image_views.into_iter().flatten());
	let texture_count = write.elements().len();
	log::debug!("texture count (variable descriptor count): {}", texture_count);
	let textures_set = PersistentDescriptorSet::new_variable(set_alloc, set_layout, texture_count, [write], [])?;

	Ok((textures_set, mat_tex_base_indices))
}

struct ModelInstance
{
	material_variant: usize,
	affine: DAffine3,
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
			affine: Default::default(),
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
	descriptor_set_allocator: StandardDescriptorSetAllocator,
	material_textures_set_layout: Arc<DescriptorSetLayout>,

	material_pipelines: BTreeMap<TypeId, MaterialPipelines>,

	// Loaded 3D models, with the key being the path relative to the current working directory.
	// Each model will also contain the model instance for each entity.
	models: HashMap<PathBuf, Model>,

	// A mapping between entity IDs and the model it uses.
	resources: HashMap<EntityId, PathBuf>,

	cb_3d: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,
}
impl MeshManager
{
	pub fn new(render_ctx: &mut RenderContext, light_set_layout: Arc<DescriptorSetLayout>) -> crate::Result<Self>
	{
		let vk_dev = render_ctx.memory_allocator.device().clone();
		let set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 128, // we might eventually need more than this
			..Default::default()
		};
		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(vk_dev.clone(), set_alloc_info);

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

		/*render_ctx.load_transparency(material_textures_set_layout.clone())?;
		let transparency_input_layout = render_ctx
		.transparency_renderer
		.as_ref()
		.unwrap()
		.get_moments_images_set()
		.layout()
		.clone();*/

		let push_constant_size = std::mem::size_of::<Mat4>() + std::mem::size_of::<Vec4>() * 3;
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

		/*let layout_info_oit = PipelineLayoutCreateInfo {
			set_layouts: vec![
				material_textures_set_layout.clone(),
				light_set_layout,
				//transparency_input_layout,
			],
			push_constant_ranges: vec![push_constant_range],
			..Default::default()
		};
		let pipeline_layout_oit = PipelineLayout::new(vk_dev.clone(), layout_info_oit)?;*/

		// Load all registered material pipelines
		let mut material_pipelines = BTreeMap::new();
		for conf in inventory::iter::<MaterialPipelineConfig> {
			let type_id = (conf.type_id)();
			log::debug!("loading material pipeline '{}' ({:?})", conf.name, type_id);

			let pipeline_data = conf.into_pipelines(
				render_ctx.rasterization_samples,
				render_ctx.depth_stencil_format,
				pipeline_layout.clone(),
				//pipeline_layout_oit.clone(),
			)?;

			material_pipelines.insert(type_id, pipeline_data);
		}

		Ok(Self {
			descriptor_set_allocator,
			material_textures_set_layout,
			material_pipelines,
			models: Default::default(),
			resources: Default::default(),
			cb_3d: Default::default(),
		})
	}

	/// Load the model for the given `Mesh`.
	pub fn load(&mut self, render_ctx: &mut RenderContext, eid: EntityId, component: &Mesh) -> crate::Result<()>
	{
		// Get a 3D model from `path`, relative to the current working directory. This attempts
		// loading if it hasn't been loaded into memory yet.
		let loaded_model = match self.models.get_mut(&component.model_path) {
			Some(m) => m,
			None => {
				let new_model = Model::new(
					render_ctx,
					&self.descriptor_set_allocator,
					self.material_textures_set_layout.clone(),
					&component.model_path,
				)?;
				self.models.insert(component.model_path.clone(), new_model);
				self.models.get_mut(&component.model_path).unwrap()
			}
		};

		loaded_model.new_user(eid, component.material_variant.clone());
		self.resources.insert(eid, component.model_path.clone());

		Ok(())
	}

	pub fn set_affine(&mut self, eid: EntityId, affine: DAffine3)
	{
		let path = self.resources.get(&eid).unwrap().as_path();
		self.models.get_mut(path).unwrap().set_affine(eid, affine);
	}

	/// Free resources for the given entity ID. Only call this when the `Mesh` component was actually removed!
	pub fn cleanup_removed(&mut self, eid: EntityId)
	{
		let path = self.resources.get(&eid).unwrap().as_path();
		self.models.get_mut(path).unwrap().cleanup(eid);
		self.resources.remove(&eid);
	}

	fn draw(
		&self,
		render_ctx: &RenderContext,
		projview: DMat4,
		pass_type: PassType,
		common_sets: &[Arc<PersistentDescriptorSet>],
	) -> crate::Result<Option<Arc<SecondaryAutoCommandBuffer>>>
	{
		let (depth_format, viewport_extent, shadow_pass) = match &pass_type {
			PassType::Shadow(light_manager) => {
				let format = light_manager.get_dir_light_shadow().format();
				let dir_light_extent = light_manager.get_dir_light_shadow().image().extent();
				let viewport_extent = [dir_light_extent[0], dir_light_extent[1]];
				(format, viewport_extent, true)
			}
			_ => (render_ctx.depth_stencil_format, render_ctx.window_dimensions(), false),
		};

		let rasterization_samples = if !shadow_pass {
			render_ctx.rasterization_samples
		} else {
			SampleCount::Sample1
		};
		let rendering_inheritance = CommandBufferInheritanceRenderingInfo {
			color_attachment_formats: pass_type.render_color_formats(),
			depth_attachment_format: Some(depth_format),
			stencil_attachment_format: matches!(pass_type, PassType::Transparency).then_some(depth_format),
			rasterization_samples,
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
		//let transparency_pass = matches!(pass_type, PassType::TransparencyMoments(_) | PassType::Transparency);
		let transparency_pass = matches!(pass_type, PassType::Transparency);

		let mut any_drawn = false;
		for (pipeline_id, mat_pl) in &self.material_pipelines {
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

			if !common_sets.is_empty() {
				let sets = Vec::from(common_sets);
				cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 1, sets)?;
			}

			// don't filter by material pipeline ID if there is a pipeline override
			let pipeline_id_option = pipeline_override.is_none().then_some(pipeline_id);

			for model in self.models.values() {
				if model.draw(
					&mut cb,
					pipeline_id_option.copied(),
					pipeline_layout.clone(),
					transparency_pass,
					shadow_pass,
					&projview,
				) {
					any_drawn = true;
				}
			}

			if pipeline_override.is_some() {
				break;
			}
		}

		// If this is the opaque pass, don't let the command buffer leave this `MeshManager`.
		// Otherwise, only return the command buffer if this is the OIT pass and models were drawn
		// at all, or if this is the shadow pass.
		match pass_type {
			PassType::Opaque => {
				*self.cb_3d.lock().unwrap() = Some(cb.build()?);
			}
			/*PassType::TransparencyMoments(_) |*/
			PassType::Transparency => {
				if any_drawn {
					return Ok(Some(cb.build()?));
				}
			}
			PassType::Shadow(light_manager) => {
				light_manager.add_dir_light_cb(cb.build()?);
			}
		}
		Ok(None)
	}

	/// Execute the secondary command buffers for rendering the opaque 3D models.
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
				store_op: AttachmentStoreOp::Store, // OIT needs this to be `Store`
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

enum PassType<'a>
{
	Shadow(&'a LightManager),
	Opaque,
	//TransparencyMoments(Arc<GraphicsPipeline>),
	Transparency,
}
impl PassType<'_>
{
	fn render_color_formats(&self) -> Vec<Option<Format>>
	{
		let formats: &'static [Format] = match self {
			PassType::Shadow { .. } => &[],
			PassType::Opaque => &[Format::R16G16B16A16_SFLOAT],
			/*PassType::TransparencyMoments(_) => {
				&[Format::R32G32B32A32_SFLOAT, Format::R32_SFLOAT /*, Format::R32_SFLOAT*/]
			}*/
			PassType::Transparency => &[Format::R16G16B16A16_SFLOAT, Format::R8_UNORM],
		};
		formats.iter().copied().map(Some).collect()
	}
	fn pipeline(&self) -> Option<&Arc<GraphicsPipeline>>
	{
		match self {
			PassType::Shadow(light_manager) => Some(light_manager.get_shadow_pipeline()),
			//PassType::TransparencyMoments(pipeline) => Some(pipeline),
			_ => None,
		}
	}
}

//
/* Workloads and systems for drawing models */
//

// Render shadow maps.
pub(crate) fn draw_shadows(
	render_ctx: UniqueView<RenderContext>,
	mesh_manager: UniqueView<MeshManager>,
	light_manager: UniqueView<LightManager>,
) -> crate::Result<()>
{
	for layer_projview in light_manager.get_dir_light_projviews() {
		mesh_manager.draw(&render_ctx, layer_projview, PassType::Shadow(&light_manager), &[])?;
	}
	Ok(())
}

// Draw opaque 3D objects.
pub(crate) fn draw_3d(
	render_ctx: UniqueView<RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	mesh_manager: UniqueView<MeshManager>,
	light_manager: UniqueView<LightManager>,
) -> crate::Result<()>
{
	let common_sets = [light_manager.get_all_lights_set().clone()];
	mesh_manager.draw(&render_ctx, camera_manager.projview(), PassType::Opaque, &common_sets)?;
	Ok(())
}

// Draw objects for OIT (order-independent transparency).
pub(crate) fn draw_3d_oit(
	render_ctx: UniqueView<RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	mesh_manager: UniqueView<MeshManager>,
	light_manager: UniqueView<LightManager>,
) -> crate::Result<()>
{
	/* for Moment Transparency */
	/*// We do both passes for OIT in this function, because there will almost always be fewer draw
	// calls for transparent objects.
	if let Some(transparency_renderer) = &render_ctx.transparency_renderer {
		// First, collect moments for Moment Transparency (OIT). This will bind the pipeline for us,
		// since it doesn't need to do anything specific to materials (it only reads the alpha
		// channel of each texture).
		let projview = camera_manager.projview();
		let moments_pass = PassType::TransparencyMoments(transparency_renderer.get_moments_pipeline().clone());
		let moments_cb = mesh_manager.draw(&render_ctx, projview, moments_pass, &[])?;

		// Next, do the weights pass for OIT.
		if let Some(some_moments_cb) = moments_cb {
			let common_sets = [
				light_manager.get_all_lights_set().clone(),
				transparency_renderer.get_moments_images_set().clone(),
			];
			let weights_cb = mesh_manager.draw(&render_ctx, projview, PassType::Transparency, &common_sets)?;
			transparency_renderer.add_transparency_cb(some_moments_cb, weights_cb.unwrap());
		}
	}*/

	/* for WBOIT */
	let projview = camera_manager.projview();
	let common_sets = [light_manager.get_all_lights_set().clone()];
	let weights_cb = mesh_manager.draw(&render_ctx, projview, PassType::Transparency, &common_sets)?;
	if let Some(some_weights_cb) = weights_cb {
		render_ctx.transparency_renderer.add_transparency_cb(some_weights_cb);
	}

	Ok(())
}
