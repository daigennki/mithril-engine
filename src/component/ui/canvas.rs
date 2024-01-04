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
	AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo, SecondaryAutoCommandBuffer,
	SubpassContents,
};
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{
	sampler::{Filter, Sampler, SamplerCreateInfo},
	view::ImageView,
	ImageFormatInfo, ImageUsage,
};
use vulkano::pipeline::{
	graphics::{
		color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
		input_assembly::PrimitiveTopology,
		rasterization::RasterizationState,
		subpass::PipelineRenderingCreateInfo,
		GraphicsPipeline,
	},
	layout::{PipelineLayoutCreateInfo, PushConstantRange},
	Pipeline, PipelineBindPoint, PipelineLayout,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

use super::mesh::MeshType;
use crate::render::RenderContext;

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

			layout(location = 0) in vec2 pos;
			layout(location = 1) in vec2 uv;
			layout(location = 0) out vec2 texcoord;

			void main()
			{
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

			// for position input, xy: top left pos, zw: width and height
			layout(location = 0) in vec4 pos_INSTANCE;
			layout(location = 1) in vec4 color_INSTANCE;
			layout(location = 0) out vec2 texcoord;
			layout(location = 1) flat out int instance_index;
			layout(location = 2) flat out vec4 glyph_color;

			void main()
			{
				vec4 pos = pos_INSTANCE;
				vec2 texcoords[4] = { { 0.0, 0.0 }, { 0.0, pos.w }, { pos.z, 0.0 }, pos.zw };
				vec2 texcoord_logical_pixels = texcoords[min(gl_VertexIndex, 3)];
				vec2 pos2 = texcoord_logical_pixels + pos.xy;
				texcoord = texcoord_logical_pixels * texture_size_inv;

				gl_Position = vec4(transformation * pos2 + translation_projected, 0.0, 1.0);

				instance_index = gl_InstanceIndex;
				glyph_color = color_INSTANCE;
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

struct UiGpuResources
{
	text_vbo: Option<Subbuffer<[Vec4]>>,
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

	set_layout: Arc<DescriptorSetLayout>,
	text_set_layout: Arc<DescriptorSetLayout>,
	ui_pipeline: Arc<GraphicsPipeline>,
	text_pipeline: Arc<GraphicsPipeline>,

	text_resources: BTreeMap<EntityId, UiGpuResources>,
	mesh_resources: BTreeMap<EntityId, UiGpuResources>,

	quad_pos_buf: Subbuffer<[Vec2]>,
	quad_uv_buf: Subbuffer<[Vec2]>,

	default_font: Font<'static>,

	ui_cb: Option<Arc<SecondaryAutoCommandBuffer>>,
}
impl Canvas
{
	pub fn new(render_ctx: &mut RenderContext, canvas_width: u32, canvas_height: u32) -> crate::Result<Self>
	{
		let device = render_ctx.descriptor_set_allocator().device().clone();

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
			bindings: [(0, image_binding)].into(),
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

		let blend_attachment_state = ColorBlendAttachmentState {
			blend: Some(AttachmentBlend::alpha()),
			..Default::default()
		};
		let color_blend_state = ColorBlendState::with_attachment_states(1, blend_attachment_state);

		let rendering_formats = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
			..Default::default()
		};

		let ui_pipeline = crate::render::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleStrip,
			&[ui_vs::load(device.clone())?, ui_fs::load(device.clone())?],
			RasterizationState::default(),
			pipeline_layout,
			rendering_formats.clone(),
			Some(color_blend_state.clone()),
			None,
		)?;

		/* UI text pipeline */
		let text_sampler = Sampler::new(device.clone(), SamplerCreateInfo::default())?;
		let text_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			immutable_samplers: vec![text_sampler],
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
		};
		let text_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [(0, text_binding)].into(),
			..Default::default()
		};
		let text_set_layout = DescriptorSetLayout::new(device.clone(), text_set_layout_info)?;

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

		let text_pipeline = crate::render::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleStrip,
			&[ui_text_vs::load(device.clone())?, ui_text_fs::load(device.clone())?],
			RasterizationState::default(),
			text_pipeline_layout,
			rendering_formats,
			Some(color_blend_state),
			None,
		)?;

		let quad_verts = [
			// position
			Vec2::new(-0.5, -0.5),
			Vec2::new(-0.5, 0.5),
			Vec2::new(0.5, -0.5),
			Vec2::new(0.5, 0.5),
			// texcoord
			Vec2::new(0.0, 0.0),
			Vec2::new(0.0, 1.0),
			Vec2::new(1.0, 0.0),
			Vec2::new(1.0, 1.0),
		];
		let vert_buf = render_ctx.new_buffer(&quad_verts, BufferUsage::VERTEX_BUFFER)?;
		let (quad_pos_buf, quad_uv_buf) = vert_buf.split_at(4);

		let font_data = include_bytes!("../../../resource/mplus-1m-medium.ttf");
		let default_font = Font::try_from_bytes(font_data as &[u8]).ok_or("Font has invalid data")?;

		let dim = render_ctx.swapchain_dimensions();

		let (canvas_scaling, scale_factor) =
			calculate_projection(canvas_width as f32, canvas_height as f32, dim[0] as f32, dim[1] as f32);

		Ok(Canvas {
			base_dimensions: [canvas_width, canvas_height],
			canvas_scaling,
			scale_factor,
			set_layout,
			text_set_layout,
			ui_pipeline,
			text_pipeline,
			text_resources: Default::default(),
			mesh_resources: Default::default(),
			quad_pos_buf,
			quad_uv_buf,
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
		render_ctx: &mut RenderContext,
		set_layout: Arc<DescriptorSetLayout>,
		transform: &super::UITransform,
		image_view: Arc<ImageView>,
		default_scale: Vec2,
		text_vbo: Option<Subbuffer<[Vec4]>>,
	) -> crate::Result<UiGpuResources>
	{
		let scale = transform.scale.unwrap_or(default_scale) * self.canvas_scaling;
		let translation = transform.position.as_vec2() * self.canvas_scaling;
		let projected = Affine2::from_scale_angle_translation(scale, 0.0, translation);

		let image_extent = image_view.image().extent();

		// division inverted here so that division isn't performed in shader
		let logical_texture_size_inv = Vec2::splat(self.scale_factor) / UVec2::new(image_extent[0], image_extent[1]).as_vec2();

		let writes = [WriteDescriptorSet::image_view(0, image_view)];
		let descriptor_set = PersistentDescriptorSet::new(render_ctx.descriptor_set_allocator(), set_layout, writes, [])?;

		Ok(UiGpuResources {
			text_vbo,
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
		transform: &super::UITransform,
		mesh: &super::mesh::Mesh,
	) -> crate::Result<()>
	{
		if !mesh.image_path.as_os_str().is_empty() {
			let tex = render_ctx.get_texture(&mesh.image_path)?;
			let image_dimensions = UVec2::from(tex.dimensions()).as_vec2();
			let resources = self.update_transform(
				render_ctx,
				self.set_layout.clone(),
				transform,
				tex.view().clone(),
				image_dimensions,
				None,
			)?;
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
		transform: &super::UITransform,
		text: &super::text::UIText,
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
				resources.text_vbo = None;
			}
			return Ok(());
		}

		let glyph_image_array = text_to_image_array(text_str, &self.default_font, text.size * self.scale_factor);

		// If no visible glyphs were produced (e.g. the string was empty, or it only has space characters),
		// remove the vertex buffers from the GPU resources, and then return immediately.
		if glyph_image_array.is_empty() {
			if let Some(resources) = self.text_resources.get_mut(&eid) {
				resources.text_vbo = None;
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
			.and_then(|res| res.text_vbo.as_ref())
			.map(|vbo| (vbo.len() / 2) as usize)
			.unwrap_or(0);

		let mut combined_images = Vec::with_capacity((img_dim[0] * img_dim[1]) as usize * glyph_count);
		let mut text_pos_verts = Vec::with_capacity(glyph_count);
		for (image, tl, bb_size) in glyph_image_array.into_iter() {
			combined_images.extend_from_slice(image.into_raw().as_slice());

			let top_left_corner: Vec2 = tl / self.scale_factor;
			let logical_quad_size: Vec2 = bb_size.as_vec2() / self.scale_factor;
			text_pos_verts.push((top_left_corner, logical_quad_size).into());
		}

		let layer_count = glyph_count.try_into().unwrap();
		let tex = render_ctx.new_texture_from_slice(&combined_images, Format::R8_UNORM, img_dim, 1, layer_count)?;

		let colors = vec![text.color; glyph_count];

		// Combine position and color data into a single vertex buffer.
		let mut vbo_data = text_pos_verts;
		vbo_data.extend_from_slice(&colors);

		let text_vbo = if glyph_count == prev_glyph_count {
			// Reuse the vertex buffer if the glyph count hasn't changed.
			// If `prev_glyph_count` is greater than 0, we already know that `text_resources` for
			// the given `eid` is `Some`, so we use `unwrap` here.
			let resources = self.text_resources.get(&eid).unwrap();
			let some_vbo = resources.text_vbo.clone().unwrap();
			render_ctx.update_buffer(&vbo_data, some_vbo.clone());
			some_vbo
		} else {
			render_ctx.new_buffer(&vbo_data, BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST)?
		};

		let resources = self.update_transform(
			render_ctx,
			self.text_set_layout.clone(),
			transform,
			tex.view().clone(),
			Vec2::ONE,
			Some(text_vbo),
		)?;
		self.text_resources.insert(eid, resources);

		Ok(())
	}

	pub fn draw(&mut self, render_ctx: &RenderContext) -> crate::Result<()>
	{
		// TODO: how do we respect the render order of each UI element?

		let vp_extent = render_ctx.swapchain_dimensions();
		let mut cb = render_ctx.gather_commands(&[Format::R16G16B16A16_SFLOAT], None, None, vp_extent)?;

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
					cb.bind_vertex_buffers(0, (self.quad_pos_buf.clone(), self.quad_uv_buf.clone()))?;
					cb.draw(4, 1, 0, 0)?;
				}
				MeshType::Frame(_border_width) => {
					todo!();
				}
			}
		}

		cb.bind_pipeline_graphics(self.text_pipeline.clone())?;
		for resources in self.text_resources.values() {
			if let Some(vbo) = resources.text_vbo.clone() {
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
				let glyph_count = vbo.len() / 2;
				cb.bind_vertex_buffers(0, vbo.clone().split_at(glyph_count))?;
				cb.draw(4, glyph_count as u32, 0, 0)?;
			}
		}

		assert!(self.ui_cb.replace(cb.build()?).is_none());

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
	let max_width: u32 = glyphs
		.iter()
		.filter_map(|glyph| glyph.pixel_bounding_box())
		.map(|bb| bb.width().abs() as u32)
		.max()
		.unwrap_or(1)
		.next_multiple_of(8);
	let max_height: u32 = glyphs
		.iter()
		.filter_map(|glyph| glyph.pixel_bounding_box())
		.map(|bb| bb.height().abs() as u32)
		.max()
		.unwrap_or(1)
		.next_multiple_of(8);

	let mut bitmaps = Vec::with_capacity(glyphs.len());
	for glyph in glyphs {
		if let Some(bb) = glyph.pixel_bounding_box() {
			let mut image = DynamicImage::new_luma8(max_width, max_height).into_luma8();

			// Draw the glyph into the image per-pixel by using the draw closure,
			// and turn the coverage into an alpha value
			glyph.draw(|x, y, v| image.put_pixel(x, y, Luma([(v * 255.0) as u8])));

			let pos_tl = Vec2::new(bb.min.x as f32, bb.min.y as f32);
			let bounding_box_size = IVec2::new(bb.max.x - bb.min.x, bb.max.y - bb.min.y);

			bitmaps.push((image, pos_tl, bounding_box_size));
		}
	}

	bitmaps
}
