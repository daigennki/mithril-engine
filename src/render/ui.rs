/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use image::{DynamicImage, GrayImage, Luma};
use rusttype::{Font, Scale};
use shipyard::EntityId;
use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderingInfo, CommandBufferUsage,
	PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo, SecondaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::{
	allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{
	sampler::{Filter, Sampler, SamplerCreateInfo},
	view::ImageView,
	ImageCreateInfo, ImageFormatInfo, ImageUsage,
};
use vulkano::pipeline::{
	graphics::{
		color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
		input_assembly::{InputAssemblyState, PrimitiveTopology},
		subpass::PipelineRenderingCreateInfo,
		vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState},
		viewport::Viewport,
		GraphicsPipeline, GraphicsPipelineCreateInfo,
	},
	layout::{PipelineLayoutCreateInfo, PushConstantRange},
	DynamicState, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

use super::RenderContext;
use crate::component::ui::{
	mesh::{Mesh, MeshType},
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
	pos: Vec4,
	color: Vec4,
}
struct UiGpuResources
{
	glyph_info_buffer: Option<Subbuffer<[GlyphInfo]>>,
	descriptor_set: Arc<PersistentDescriptorSet>,
	projected: Affine2,
	logical_texture_size_inv: Vec2,
	mesh_type: MeshType,
}

#[derive(shipyard::Unique)]
pub struct Canvas
{
	base_dimensions: [u32; 2],
	canvas_scaling: Vec2,
	scale_factor: f32,

	descriptor_set_allocator: StandardDescriptorSetAllocator,
	set_layout: Arc<DescriptorSetLayout>,
	text_set_layout: Arc<DescriptorSetLayout>,
	ui_pipeline: Arc<GraphicsPipeline>,
	text_pipeline: Arc<GraphicsPipeline>,

	text_resources: BTreeMap<EntityId, UiGpuResources>,
	mesh_resources: BTreeMap<EntityId, UiGpuResources>,

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

		let push_constant_size = std::mem::size_of::<Mat2>() + std::mem::size_of::<Vec2>();
		let pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![set_layout.clone()],
			push_constant_ranges: vec![PushConstantRange {
				stages: ShaderStages::VERTEX,
				offset: 0,
				size: push_constant_size.try_into().unwrap(),
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

		let text_push_constant_size = std::mem::size_of::<Mat2>() + std::mem::size_of::<Vec2>() * 2;
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

		let quad_verts = vec![
			// position (xy) and texcoord (zw)
			Vec4::new(-0.5, -0.5, 0.0, 0.0),
			Vec4::new(-0.5, 0.5, 0.0, 1.0),
			Vec4::new(0.5, -0.5, 1.0, 0.0),
			Vec4::new(0.5, 0.5, 1.0, 1.0),
		];
		let quad_vbo = render_ctx.new_buffer(quad_verts, BufferUsage::VERTEX_BUFFER)?;

		let font_data = include_bytes!("../../resource/mplus-1m-medium.ttf");
		let default_font = Font::try_from_bytes(font_data as &[u8]).ok_or("Font has invalid data")?;

		let dim = render_ctx.swapchain_dimensions();

		let (canvas_scaling, scale_factor) =
			calculate_projection(canvas_width as f32, canvas_height as f32, dim[0] as f32, dim[1] as f32);

		Ok(Canvas {
			base_dimensions: [canvas_width, canvas_height],
			canvas_scaling,
			scale_factor,
			descriptor_set_allocator,
			set_layout,
			text_set_layout,
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

	fn update_transform(
		&mut self,
		set_layout: Arc<DescriptorSetLayout>,
		transform: &UITransform,
		image_view: Arc<ImageView>,
		default_scale: Vec2,
		glyph_info_buffer: Option<Subbuffer<[GlyphInfo]>>,
	) -> crate::Result<UiGpuResources>
	{
		let scale = transform.scale.unwrap_or(default_scale) * self.canvas_scaling;
		let translation = transform.position.as_vec2() * self.canvas_scaling;
		let projected = Affine2::from_scale_angle_translation(scale, 0.0, translation);

		let image_extent = image_view.image().extent();

		// division inverted here so that division isn't performed in shader
		let logical_texture_size_inv = Vec2::splat(self.scale_factor) / UVec2::new(image_extent[0], image_extent[1]).as_vec2();

		let writes: smallvec::SmallVec<[WriteDescriptorSet; 2]> = if let Some(buf) = glyph_info_buffer.clone() {
			smallvec::smallvec![
				WriteDescriptorSet::image_view(0, image_view),
				WriteDescriptorSet::buffer(1, buf),
			]
		} else {
			smallvec::smallvec![WriteDescriptorSet::image_view(0, image_view)]
		};
		let descriptor_set = PersistentDescriptorSet::new(&self.descriptor_set_allocator, set_layout, writes.into_iter(), [])?;

		Ok(UiGpuResources {
			glyph_info_buffer,
			descriptor_set,
			projected,
			logical_texture_size_inv,
			mesh_type: MeshType::Quad,
		})
	}

	/// Update the GPU resources for entities with a `Mesh` component.
	/// Call this whenever the component has been inserted or modified.
	pub fn update_mesh(
		&mut self,
		render_ctx: &mut RenderContext,
		eid: EntityId,
		transform: &UITransform,
		mesh: &Mesh,
	) -> crate::Result<()>
	{
		if !mesh.image_path.as_os_str().is_empty() {
			let tex = render_ctx.new_texture(&mesh.image_path)?;
			let image_extent = tex.image().extent();
			let image_dimensions = Vec2::new(image_extent[0] as f32, image_extent[1] as f32);
			let resources = self.update_transform(self.set_layout.clone(), transform, tex, image_dimensions, None)?;
			self.mesh_resources.insert(eid, resources);
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
		let text_str = &text.text_str;

		// If the string is longer than the maximum image array layers allowed by Vulkan, refuse to
		// render it. On Windows/Linux desktop, the limit is always at least 2048, so it should be
		// extremely rare that we get such a long string, but we should check it anyways just in case.
		// We also check that it's no larger than 2048 characters because we might use
		// vkCmdUpdateBuffer, which has a limit of 65536 (32 * 2048) bytes, to update vertices.
		let image_format_info = ImageFormatInfo {
			format: Format::R8_UNORM,
			usage: ImageUsage::SAMPLED,
			..Default::default()
		};
		let max_glyphs: usize = render_ctx
			.memory_allocator
			.device()
			.physical_device()
			.image_format_properties(image_format_info)?
			.ok_or("text image format is not supported by the physical device")?
			.max_array_layers
			.min(2048)
			.try_into()
			.unwrap();
		if text_str.chars().count() >= max_glyphs {
			log::warn!(
				"UI text string too long ({} chars, limit is {})! Refusing to render string: {}",
				text_str.len(),
				max_glyphs,
				text_str,
			);
			if let Some(resources) = self.text_resources.get_mut(&eid) {
				resources.glyph_info_buffer = None;
			}
			return Ok(());
		}

		let glyph_image_array = text_to_image_array(text_str, &self.default_font, text.size * self.scale_factor);

		// If no visible glyphs were produced (e.g. the string was empty, or it only has space characters),
		// remove the vertex buffers from the GPU resources, and then return immediately.
		if glyph_image_array.is_empty() {
			if let Some(resources) = self.text_resources.get_mut(&eid) {
				resources.glyph_info_buffer = None;
			}
			return Ok(());
		}

		let img_dim = glyph_image_array
			.first()
			.map(|(image, _, _)| [image.width(), image.height()])
			.unwrap();

		let glyph_count = glyph_image_array.len();
		let prev_glyph_count = self
			.text_resources
			.get(&eid)
			.as_ref()
			.and_then(|res| res.glyph_info_buffer.as_ref())
			.map(|buf| buf.len() as usize)
			.unwrap_or(0);

		let mut combined_images = Vec::with_capacity((img_dim[0] * img_dim[1]) as usize * glyph_count);
		let mut text_pos_verts = Vec::with_capacity(glyph_count);
		for (image, tl, bb_size) in glyph_image_array.into_iter() {
			combined_images.extend_from_slice(image.into_raw().as_slice());

			let top_left_corner: Vec2 = tl / self.scale_factor;
			let logical_quad_size: Vec2 = bb_size.as_vec2() / self.scale_factor;
			text_pos_verts.push((top_left_corner, logical_quad_size).into());
		}

		let image_create_info = ImageCreateInfo {
			format: Format::R8_UNORM,
			extent: [img_dim[0], img_dim[1], 1],
			array_layers: glyph_count.try_into().unwrap(),
			usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
			..Default::default()
		};
		let tex_image = render_ctx.new_image(combined_images, image_create_info)?;
		let tex = ImageView::new_default(tex_image)?;

		let colors = vec![text.color; glyph_count];

		// Combine position and color data into a single storage buffer.
		let glyph_infos: Vec<_> = text_pos_verts
			.into_iter()
			.zip(colors)
			.map(|(pos, color)| GlyphInfo { pos, color })
			.collect();

		let glyph_info_buf = if glyph_count == prev_glyph_count {
			// Reuse the buffer if the glyph count hasn't changed.
			// If `prev_glyph_count` is greater than 0, we already know that `text_resources` for
			// the given `eid` is `Some`, so we use `unwrap` here.
			let resources = self.text_resources.get(&eid).unwrap();
			let some_buf = resources.glyph_info_buffer.clone().unwrap();
			render_ctx
				.transfer_manager
				.update_buffer(glyph_infos.into(), some_buf.clone());
			some_buf
		} else {
			render_ctx.new_buffer(glyph_infos, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST)?
		};

		let resources = self.update_transform(self.text_set_layout.clone(), transform, tex, Vec2::ONE, Some(glyph_info_buf))?;
		self.text_resources.insert(eid, resources);

		Ok(())
	}

	pub fn draw(&mut self, render_ctx: &RenderContext) -> crate::Result<()>
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

		let vp_extent = render_ctx.swapchain_dimensions();
		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [vp_extent[0] as f32, vp_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};
		cb.set_viewport(0, [viewport].as_slice().into())?;

		cb.bind_pipeline_graphics(self.ui_pipeline.clone())?;
		for resources in self.mesh_resources.values() {
			cb.push_constants(self.ui_pipeline.layout().clone(), 0, resources.projected)?;
			cb.bind_descriptor_sets(
				PipelineBindPoint::Graphics,
				self.ui_pipeline.layout().clone(),
				0,
				vec![resources.descriptor_set.clone()],
			)?;

			match resources.mesh_type {
				MeshType::Quad => {
					cb.bind_vertex_buffers(0, (self.quad_vbo.clone(),))?;
					cb.draw(4, 1, 0, 0)?;
				}
				MeshType::Frame(_border_width) => {
					todo!();
				}
			}
		}

		cb.bind_pipeline_graphics(self.text_pipeline.clone())?;
		for resources in self.text_resources.values() {
			if let Some(glyph_buf) = &resources.glyph_info_buffer {
				let mut push_constant_data: [f32; 8] = Default::default();
				resources.projected.write_cols_to_slice(&mut push_constant_data[0..6]);
				resources
					.logical_texture_size_inv
					.write_to_slice(&mut push_constant_data[6..8]);

				cb.push_constants(self.text_pipeline.layout().clone(), 0, push_constant_data)?;
				cb.bind_descriptor_sets(
					PipelineBindPoint::Graphics,
					self.text_pipeline.layout().clone(),
					0,
					vec![resources.descriptor_set.clone()],
				)?;
				let glyph_count = glyph_buf.len();
				cb.draw(4, glyph_count as u32, 0, 0)?;
			}
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

/// Create a `Vec` of greyscale images for each rendered glyph.
/// Each image is paired with a `Vec2` representing the top left corner relative to the baseline,
/// and an `IVec2` representing the bounding box size.
fn text_to_image_array(text: &str, font: &Font<'static>, size: f32) -> Vec<(GrayImage, Vec2, IVec2)>
{
	let scale_uniform = Scale::uniform(size);
	let glyphs: Vec<_> = font.layout(text, scale_uniform, rusttype::point(0.0, 0.0)).collect();

	// Get the largest glyphs in terms of width and height respectively.
	// They get adjusted to the next multiple of 8 for memory alignment purposes.
	let max_width = glyphs
		.iter()
		.filter_map(|glyph| glyph.pixel_bounding_box())
		.map(|bb| bb.width().unsigned_abs())
		.max()
		.unwrap_or(1)
		.next_multiple_of(8);
	let max_height = glyphs
		.iter()
		.filter_map(|glyph| glyph.pixel_bounding_box())
		.map(|bb| bb.height().unsigned_abs())
		.max()
		.unwrap_or(1)
		.next_multiple_of(8);

	glyphs
		.into_iter()
		.filter_map(|glyph| glyph.pixel_bounding_box().map(|bb| (glyph, bb)))
		.map(|(glyph, bb)| {
			let mut image = DynamicImage::new_luma8(max_width, max_height).into_luma8();

			// Draw the glyph into the image per-pixel by using the draw closure,
			// and turn the coverage into an alpha value
			glyph.draw(|x, y, v| image.put_pixel(x, y, Luma([(v * 255.0) as u8])));

			let pos_tl = Vec2::new(bb.min.x as f32, bb.min.y as f32);
			let bounding_box_size = IVec2::new(bb.max.x - bb.min.x, bb.max.y - bb.min.y);

			(image, pos_tl, bounding_box_size)
		})
		.collect()
}
