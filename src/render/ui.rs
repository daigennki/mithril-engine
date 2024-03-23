/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use glam::*;
use image::{DynamicImage, GrayImage, Luma};
use rusttype::{Font, Scale};
use shipyard::{EntityId, UniqueView, UniqueViewMut};
use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::*;
use vulkano::descriptor_set::{allocator::*, layout::*, *};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{sampler::*, view::ImageView, ImageCreateInfo, ImageUsage};
use vulkano::pipeline::graphics::{
	color_blend::*, input_assembly::*, subpass::PipelineRenderingCreateInfo, vertex_input::*, viewport::Viewport, *,
};
use vulkano::pipeline::{layout::*, *};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

use super::RenderContext;
use crate::component::ui::{
	mesh::{MeshType, UIMesh},
	text::UIText,
	UITransform,
};

mod ui_vs
{
	vulkano_shaders::shader! {
		ty: "vertex",
		src: r"
			#version 460

			layout(push_constant) uniform pc
			{
				mat2 transformation;
				vec2 translation_projected;
			};

			layout(location = 0) in vec4 pos_and_texcoord;
			layout(location = 0) out vec2 texcoord;

			void main()
			{
				vec2 pos = pos_and_texcoord.xy;
				vec2 uv = pos_and_texcoord.zw;
				gl_Position = vec4(transformation * pos + translation_projected, 0.0, 1.0);
				texcoord = uv;
			}
		",
	}
}
mod ui_fs
{
	vulkano_shaders::shader! {
		ty: "fragment",
		src: r"
			#version 460

			layout(binding = 0) uniform sampler2D tex;

			layout(location = 0) in vec2 texcoord;
			layout(location = 0) out vec4 color_out;

			void main()
			{
				color_out = texture(tex, texcoord);
			}
		",
	}
}
mod ui_text_vs
{
	vulkano_shaders::shader! {
		ty: "vertex",
		src: r"
			#version 460

			layout(push_constant) uniform pc
			{
				mat2 transformation;
				vec2 translation_projected;
				vec2 texture_size_inv;
			};

			struct GlyphInfo
			{
				vec4 pos; // xy: top left pos, zw: width and height
				vec4 color;
			};
			layout(binding = 1) readonly buffer glyph_info
			{
				GlyphInfo glyphs[];
			};

			layout(location = 0) out vec2 texcoord;
			layout(location = 1) flat out int instance_index;
			layout(location = 2) flat out vec4 glyph_color;

			void main()
			{
				GlyphInfo glyph_info = glyphs[gl_InstanceIndex];
				vec4 pos = glyph_info.pos;
				vec2 texcoords[4] = { { 0.0, 0.0 }, { 0.0, pos.w }, { pos.z, 0.0 }, pos.zw };
				vec2 texcoord_logical_pixels = texcoords[min(gl_VertexIndex, 3)];
				vec2 pos2 = texcoord_logical_pixels + pos.xy;
				texcoord = texcoord_logical_pixels * texture_size_inv;

				gl_Position = vec4(transformation * pos2 + translation_projected, 0.0, 1.0);

				instance_index = gl_InstanceIndex;
				glyph_color = glyph_info.color;
			}
		",
	}
}
mod ui_text_fs
{
	vulkano_shaders::shader! {
		ty: "fragment",
		src: r"
			#version 460

			layout(binding = 0) uniform sampler2DArray tex;

			layout(location = 0) in vec2 texcoord;
			layout(location = 1) flat in int instance_index;
			layout(location = 2) flat in vec4 glyph_color;
			layout(location = 0) out vec4 color_out;

			void main()
			{
				vec4 sampled_color = vec4(1.0, 1.0, 1.0, texture(tex, vec3(texcoord, instance_index)).r);
				color_out = glyph_color * sampled_color;
			}
		",
	}
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C)]
struct GlyphInfo
{
	pos_size: Vec4, // pos: xy, size: zw
	color: Vec4,
}
struct TextResources
{
	descriptor_set: Arc<PersistentDescriptorSet>,
	projected: Affine2,
	logical_texture_size_inv: Vec2,
	glyph_info_buffer: Subbuffer<[GlyphInfo]>,
	update_needed: Option<Box<[GlyphInfo]>>,
}
struct MeshResources
{
	descriptor_set: Arc<PersistentDescriptorSet>,
	projected: Affine2,
	mesh_type: MeshType,
}

#[derive(shipyard::Unique)]
pub struct Canvas
{
	base_dimensions: [u32; 2],
	canvas_scaling: Vec2,
	scale_factor: f32,

	descriptor_set_allocator: StandardDescriptorSetAllocator,
	ui_pipeline: Arc<GraphicsPipeline>,
	text_pipeline: Arc<GraphicsPipeline>,

	text_resources: BTreeMap<EntityId, TextResources>,
	mesh_resources: BTreeMap<EntityId, MeshResources>,

	quad_vbo: Subbuffer<[Vec4]>,

	default_font: Font<'static>,

	ui_cb: Option<Arc<SecondaryAutoCommandBuffer>>,
}
impl Canvas
{
	pub fn new(render_ctx: &mut RenderContext, canvas_width: u32, canvas_height: u32) -> crate::Result<Self>
	{
		let device = render_ctx.memory_allocator.device().clone();
		let set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 64,
			..Default::default()
		};
		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), set_alloc_info);

		let sampler_info = SamplerCreateInfo {
			mag_filter: Filter::Linear,
			min_filter: Filter::Linear,
			..Default::default()
		};
		let sampler = Sampler::new(device.clone(), sampler_info)?;
		let image_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			immutable_samplers: vec![sampler],
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
		};
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [(0, image_binding.clone())].into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

		let pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![set_layout.clone()],
			push_constant_ranges: vec![PushConstantRange {
				stages: ShaderStages::VERTEX,
				offset: 0,
				size: std::mem::size_of::<Affine2>().try_into().unwrap(),
			}],
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_info)?;

		let input_assembly_state = InputAssemblyState {
			topology: PrimitiveTopology::TriangleStrip,
			primitive_restart_enable: true,
			..Default::default()
		};

		let blend_attachment_state = ColorBlendAttachmentState {
			blend: Some(AttachmentBlend::alpha()),
			..Default::default()
		};
		let color_blend_state = ColorBlendState::with_attachment_states(1, blend_attachment_state);

		let vert_binding = VertexInputBindingDescription {
			stride: 16,
			input_rate: VertexInputRate::Vertex,
		};
		let vert_attribute = VertexInputAttributeDescription {
			binding: 0,
			format: Format::R32G32B32A32_SFLOAT,
			offset: 0,
		};
		let vertex_input_state = VertexInputState::new().binding(0, vert_binding).attribute(0, vert_attribute);

		let rendering_formats = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
			..Default::default()
		};
		let ui_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				PipelineShaderStageCreateInfo::new(ui_vs::load(device.clone())?.entry_point("main").unwrap()),
				PipelineShaderStageCreateInfo::new(ui_fs::load(device.clone())?.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(vertex_input_state),
			input_assembly_state: Some(input_assembly_state),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(Default::default()),
			multisample_state: Some(Default::default()),
			color_blend_state: Some(color_blend_state.clone()),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(rendering_formats.clone().into()),
			..GraphicsPipelineCreateInfo::layout(pipeline_layout)
		};
		let ui_pipeline = GraphicsPipeline::new(device.clone(), None, ui_pipeline_info)?;

		/* UI text pipeline */
		let storage_buffer_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::VERTEX,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
		};
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [(0, image_binding), (1, storage_buffer_binding)].into(),
			..Default::default()
		};
		let text_set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

		let text_push_constant_size = std::mem::size_of::<Vec2>() * 4;
		let pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![text_set_layout.clone()],
			push_constant_ranges: vec![PushConstantRange {
				stages: ShaderStages::VERTEX,
				offset: 0,
				size: text_push_constant_size.try_into().unwrap(),
			}],
			..Default::default()
		};
		let text_pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_info)?;

		let text_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				PipelineShaderStageCreateInfo::new(ui_text_vs::load(device.clone())?.entry_point("main").unwrap()),
				PipelineShaderStageCreateInfo::new(ui_text_fs::load(device.clone())?.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(Default::default()),
			input_assembly_state: Some(input_assembly_state),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(Default::default()),
			multisample_state: Some(Default::default()),
			color_blend_state: Some(color_blend_state.clone()),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(rendering_formats.clone().into()),
			..GraphicsPipelineCreateInfo::layout(text_pipeline_layout)
		};
		let text_pipeline = GraphicsPipeline::new(device, None, text_pipeline_info)?;

		let quad_verts = [
			// position (xy) and texcoord (zw)
			Vec4::new(-0.5, -0.5, 0.0, 0.0),
			Vec4::new(-0.5, 0.5, 0.0, 1.0),
			Vec4::new(0.5, -0.5, 1.0, 0.0),
			Vec4::new(0.5, 0.5, 1.0, 1.0),
		];
		let quad_vbo = render_ctx.new_buffer(&quad_verts, BufferUsage::VERTEX_BUFFER)?;

		const DEFAULT_FONT_BYTES: &[u8] = include_bytes!("../../resource/mplus-1m-medium.ttf");
		let default_font = Font::try_from_bytes(DEFAULT_FONT_BYTES).unwrap();

		let dim = render_ctx.window_dimensions();

		let (canvas_scaling, scale_factor) =
			calculate_projection(canvas_width as f32, canvas_height as f32, dim[0] as f32, dim[1] as f32);

		Ok(Canvas {
			base_dimensions: [canvas_width, canvas_height],
			canvas_scaling,
			scale_factor,
			descriptor_set_allocator,
			ui_pipeline,
			text_pipeline,
			text_resources: Default::default(),
			mesh_resources: Default::default(),
			quad_vbo,
			default_font,
			ui_cb: None,
		})
	}

	/// Clean up resources for removed `UIText` components.
	pub fn cleanup_removed_text(&mut self, eid: EntityId)
	{
		self.text_resources.remove(&eid);
	}
	/// Clean up resources for removed UI mesh components.
	pub fn cleanup_removed_mesh(&mut self, eid: EntityId)
	{
		self.mesh_resources.remove(&eid);
	}

	/// Run this function whenever the screen resizes, to adjust the canvas aspect ratio to fit.
	pub fn on_screen_resize(&mut self, screen_width: u32, screen_height: u32)
	{
		let (canvas_width, canvas_height) = (self.base_dimensions[0] as f32, self.base_dimensions[1] as f32);
		let (canvas_scaling, scale_factor) =
			calculate_projection(canvas_width, canvas_height, screen_width as f32, screen_height as f32);
		self.canvas_scaling = canvas_scaling;
		self.scale_factor = scale_factor;
	}

	/// Update the GPU resources for entities with a `Mesh` component.
	/// Call this whenever the component has been inserted or modified.
	pub fn update_mesh(
		&mut self,
		render_ctx: &mut RenderContext,
		eid: EntityId,
		transform: &UITransform,
		mesh: &UIMesh,
	) -> crate::Result<()>
	{
		if !mesh.image_path.as_os_str().is_empty() {
			let tex = render_ctx.new_texture(&mesh.image_path)?;
			let image_extent = tex.image().extent();
			let image_dimensions = Vec2::new(image_extent[0] as f32, image_extent[1] as f32);

			let scale = transform.scale.unwrap_or(image_dimensions) * self.canvas_scaling;
			let translation = transform.position.as_vec2() * self.canvas_scaling;
			let projected = Affine2::from_scale_angle_translation(scale, 0.0, translation);

			let writes = [WriteDescriptorSet::image_view(0, tex)];
			let set_layout = self.ui_pipeline.layout().set_layouts()[0].clone();
			let set = PersistentDescriptorSet::new(&self.descriptor_set_allocator, set_layout, writes, [])?;

			let resources = MeshResources {
				descriptor_set: set,
				projected,
				mesh_type: MeshType::Quad,
			};
			self.mesh_resources.insert(eid, resources);
		} else {
			self.mesh_resources.remove(&eid);
		}

		Ok(())
	}

	/// Update the GPU resources for entities with a `UIText` component.
	/// Call this whenever the component has been inserted or modified.
	pub fn update_text(
		&mut self,
		render_ctx: &mut RenderContext,
		eid: EntityId,
		transform: &UITransform,
		text: &UIText,
	) -> crate::Result<()>
	{
		let mut text_str = text.text_str.as_str();

		// Truncate strings longer than the maximum image array layers allowed by the device. That's
		// often at least 2048 which is way more than enough, but we should check it anyways just in
		// case. Strings longer than 2048 characters are also truncated because vertices are updated
		// with `vkCmdUpdateBuffer`, which has a limit of 65536 (32 * 2048) bytes.
		let max_glyphs: usize = render_ctx
			.memory_allocator
			.device()
			.physical_device()
			.properties()
			.max_image_array_layers
			.min(2048)
			.try_into()
			.unwrap();
		if text_str.chars().count() > max_glyphs {
			log::warn!(
				"UI text string is too long ({} chars, limit is {})! Truncating string: {}",
				text_str.len(),
				max_glyphs,
				text_str,
			);
			let (truncate_byte_offset, _) = text_str.char_indices().nth(max_glyphs).unwrap();
			text_str = &text_str[..truncate_byte_offset];
		}

		let (optional_glyphs_image, mut glyph_infos) =
			text_to_image(text_str, &self.default_font, text.size * self.scale_factor);

		// If no visible glyphs were produced (e.g. the string was empty, or it only has space characters),
		// remove the GPU resources for the component, and then return immediately.
		let combined_image = if let Some(image) = optional_glyphs_image {
			image
		} else {
			self.text_resources.remove(&eid);
			return Ok(());
		};

		let len_u32: u32 = glyph_infos.len().try_into().unwrap();
		let img_width = combined_image.width();
		let img_height = combined_image.height() / len_u32;
		let img_dim = [img_width, img_height];

		let glyph_count = glyph_infos.len();
		let prev_glyph_count = self
			.text_resources
			.get(&eid)
			.as_ref()
			.map(|res| usize::try_from(res.glyph_info_buffer.len()).unwrap())
			.unwrap_or(0);

		// Scale the positions and sizes according to the current UI scale factor, and set the color
		// for each glyph.
		for glyph_info in &mut glyph_infos {
			glyph_info.pos_size /= self.scale_factor;
			glyph_info.color = text.color;
		}

		let image_create_info = ImageCreateInfo {
			format: Format::R8_UNORM,
			extent: [img_dim[0], img_dim[1], 1],
			array_layers: glyph_count.try_into().unwrap(),
			usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
			..Default::default()
		};
		let tex_image = render_ctx.new_image(&combined_image.into_raw(), image_create_info)?;
		let tex = ImageView::new_default(tex_image)?;

		let update_needed;
		let glyph_info_buffer = if glyph_count == prev_glyph_count {
			// Reuse the buffer if the glyph count hasn't changed. If `prev_glyph_count` is greater
			// than 0, `text_resources` for the given `eid` must be `Some`, so we use `unwrap` here.
			let resources = self.text_resources.get_mut(&eid).unwrap();
			update_needed = Some(glyph_infos.into());
			resources.glyph_info_buffer.clone()
		} else {
			update_needed = None;
			render_ctx.new_buffer(&glyph_infos, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST)?
		};

		let writes = [
			WriteDescriptorSet::image_view(0, tex),
			WriteDescriptorSet::buffer(1, glyph_info_buffer.clone()),
		];
		let set_layout = self.text_pipeline.layout().set_layouts()[0].clone();
		let set = PersistentDescriptorSet::new(&self.descriptor_set_allocator, set_layout, writes, [])?;

		let scale = transform.scale.unwrap_or(Vec2::ONE) * self.canvas_scaling;
		let translation = transform.position.as_vec2() * self.canvas_scaling;
		let projected = Affine2::from_scale_angle_translation(scale, 0.0, translation);

		// division inverted here so that division isn't performed in shader
		let logical_texture_size_inv = Vec2::splat(self.scale_factor) / UVec2::new(img_dim[0], img_dim[1]).as_vec2();

		let resources = TextResources {
			descriptor_set: set,
			projected,
			logical_texture_size_inv,
			glyph_info_buffer,
			update_needed,
		};
		self.text_resources.insert(eid, resources);

		Ok(())
	}

	fn draw(&mut self, render_ctx: &RenderContext) -> crate::Result<()>
	{
		// TODO: how do we respect the render order of each UI element?
		let rendering_inheritance = CommandBufferInheritanceRenderingInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
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

		let vp_extent = render_ctx.window_dimensions();
		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [vp_extent[0] as f32, vp_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};
		cb.set_viewport(0, smallvec::smallvec![viewport])?
			.bind_pipeline_graphics(self.ui_pipeline.clone())?;

		let ui_pipeline_layout = self.ui_pipeline.layout().clone();
		for resources in self.mesh_resources.values() {
			let set = resources.descriptor_set.clone();
			cb.push_constants(ui_pipeline_layout.clone(), 0, resources.projected)?
				.bind_descriptor_sets(PipelineBindPoint::Graphics, ui_pipeline_layout.clone(), 0, set)?;

			match resources.mesh_type {
				MeshType::Quad => {
					cb.bind_vertex_buffers(0, self.quad_vbo.clone())?.draw(4, 1, 0, 0)?;
				}
				MeshType::Frame(_border_width) => {
					todo!();
				}
			}
		}

		cb.bind_pipeline_graphics(self.text_pipeline.clone())?;
		let text_pipeline_layout = self.text_pipeline.layout().clone();
		for resources in self.text_resources.values() {
			let mut push_constant_data: [f32; 8] = Default::default();
			resources.projected.write_cols_to_slice(&mut push_constant_data[0..6]);
			resources
				.logical_texture_size_inv
				.write_to_slice(&mut push_constant_data[6..8]);

			let set = resources.descriptor_set.clone();
			let glyph_count = resources.glyph_info_buffer.len().try_into().unwrap();
			cb.push_constants(text_pipeline_layout.clone(), 0, push_constant_data)?
				.bind_descriptor_sets(PipelineBindPoint::Graphics, text_pipeline_layout.clone(), 0, set)?
				.draw(4, glyph_count, 0, 0)?;
		}

		self.ui_cb = Some(cb.build()?);

		Ok(())
	}

	pub fn execute_rendering(
		&mut self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		color_image: Arc<ImageView>,
	) -> crate::Result<()>
	{
		for text_resource in self.text_resources.values_mut() {
			if let Some(update_needed) = text_resource.update_needed.take() {
				let dst_buf = text_resource.glyph_info_buffer.clone();
				cb_builder.update_buffer(dst_buf, update_needed)?;
			}
		}

		if let Some(some_cb) = self.ui_cb.take() {
			let ui_render_info = RenderingInfo {
				color_attachments: vec![Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Load,
					store_op: AttachmentStoreOp::Store,
					..RenderingAttachmentInfo::image_view(color_image)
				})],
				contents: SubpassContents::SecondaryCommandBuffers,
				..Default::default()
			};
			cb_builder
				.begin_rendering(ui_render_info)?
				.execute_commands(some_cb)?
				.end_rendering()?;
		}

		Ok(())
	}
}

fn calculate_projection(canvas_width: f32, canvas_height: f32, screen_width: f32, screen_height: f32) -> (Vec2, f32)
{
	let canvas_aspect_ratio = canvas_width / canvas_height;
	let screen_aspect_ratio = screen_width / screen_height;

	// UI scale factor, used to increase resolution of components such as text when necessary
	let scale_factor;

	// Adjusted canvas dimensions
	let (adjusted_canvas_w, adjusted_canvas_h);

	// If the screen is wider than the canvas, make the canvas wider.
	// Otherwise, make the canvas taller.
	if screen_aspect_ratio > canvas_aspect_ratio {
		adjusted_canvas_w = canvas_height * screen_width / screen_height;
		adjusted_canvas_h = canvas_height;
		scale_factor = screen_height / canvas_height;
	} else {
		adjusted_canvas_w = canvas_width;
		adjusted_canvas_h = canvas_width * screen_height / screen_width;
		scale_factor = screen_width / canvas_width;
	}

	let proj = 2.0 / Vec2::new(adjusted_canvas_w, adjusted_canvas_h);

	(proj, scale_factor)
}

/// Create a combined image containing each glyph in order. Divide total image height by the length
/// of the `Vec` containing glyph info to get the max height of the glyphs.
///
/// Each image is paired with a `GlyphInfo` containing a `Vec4` with xy representing the top left
/// corner relative to the baseline, and zw representing the bounding box size. The `color` will be
/// set to white, which you can change later for each glyph.
fn text_to_image(text: &str, font: &Font<'_>, size: f32) -> (Option<GrayImage>, Vec<GlyphInfo>)
{
	let scale_uniform = Scale::uniform(size);
	let glyphs: Vec<_> = font
		.layout(text, scale_uniform, rusttype::point(0.0, 0.0))
		.filter_map(|glyph| glyph.pixel_bounding_box().map(|bb| (glyph, bb)))
		.collect();

	if glyphs.is_empty() {
		return (None, Vec::new());
	}

	// Get the largest glyphs in terms of width and height respectively.
	// They get adjusted to the next multiple of 8 for memory alignment purposes.
	let max_width = glyphs
		.iter()
		.map(|(_, bb)| bb.width().unsigned_abs())
		.max()
		.unwrap()
		.next_multiple_of(8);
	let max_height = glyphs
		.iter()
		.map(|(_, bb)| bb.height().unsigned_abs())
		.max()
		.unwrap()
		.next_multiple_of(8);

	let glyphs_len_u32: u32 = glyphs.len().try_into().unwrap();
	let mut combined_images = DynamicImage::new_luma8(max_width, max_height * glyphs_len_u32).into_luma8();
	let positions_sizes = glyphs
		.into_iter()
		.zip(0..)
		.map(|((glyph, bb), i)| {
			let offset_lines = max_height * i;

			// Draw the glyph into the image per-pixel, and turn the coverage into an alpha value.
			glyph.draw(|x, y, v| combined_images.put_pixel(x, y + offset_lines, Luma([(v * 255.0) as u8])));

			let position_size_int = IVec4::new(bb.min.x, bb.min.y, bb.max.x - bb.min.x, bb.max.y - bb.min.y);
			GlyphInfo {
				pos_size: position_size_int.as_vec4(),
				color: Vec4::splat(1.0),
			}
		})
		.collect();

	(Some(combined_images), positions_sizes)
}

/* UI workload system */
pub(crate) fn draw_ui(render_ctx: UniqueView<RenderContext>, mut canvas: UniqueViewMut<Canvas>) -> crate::Result<()>
{
	canvas.draw(&render_ctx)
}
