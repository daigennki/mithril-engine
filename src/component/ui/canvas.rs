/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use image::{DynamicImage, Luma, GrayImage};
use rusttype::{Font, Scale};
use shipyard::EntityId;
use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DrawIndirectCommand, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{
	sampler::{Filter, Sampler, SamplerCreateInfo},
	view::ImageView, ImageFormatInfo, ImageUsage,
};
use vulkano::pipeline::{
	graphics::{
		color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
		input_assembly::PrimitiveTopology,
		rasterization::RasterizationState,
		subpass::PipelineRenderingCreateInfo,
		GraphicsPipeline,
	},
	Pipeline,
};
use vulkano::shader::ShaderStages;

use super::mesh::MeshType;
use crate::render::RenderContext;
use crate::GenericEngineError;

mod ui_vs
{
	vulkano_shaders::shader! {
		ty: "vertex",
		src: r"
			#version 460

			layout(binding = 0) uniform transform_ubo
			{
				mat4 transformation;
			};

			layout(location = 0) in vec2 pos;
			layout(location = 1) in vec2 uv;
			layout(location = 0) out vec2 texcoord;

			void main()
			{
				gl_Position = transformation * vec4(pos, 0.0, 1.0);
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

			layout(binding = 1) uniform sampler2D tex;

			layout(location = 0) in vec2 texcoord;
			layout(location = 0) out vec4 color_out;

			void main()
			{
				color_out = texture(tex, texcoord);
				color_out.rgb *= color_out.a;
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

			layout(binding = 0) uniform transform_ubo
			{
				mat4 transformation;
			};

			layout(location = 0) in vec2 pos;
			layout(location = 0) out vec2 texcoord;
			layout(location = 1) flat out int draw_id;

			void main()
			{
				gl_Position = transformation * vec4(pos, 0.0, 1.0);
				vec2 texcoords[4] = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };
				texcoord = texcoords[gl_VertexIndex % 4];
				draw_id = gl_DrawID;
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

			layout(binding = 1) uniform sampler2DArray tex;

			layout(location = 0) in vec2 texcoord;
			layout(location = 1) flat in int draw_id;
			layout(location = 0) out vec4 color_out;

			void main()
			{
				color_out = vec4(0.0, 0.0, 0.0, texture(tex, vec3(texcoord, draw_id)).r);
			}
		",
	}
}

struct UiGpuResources
{
	pub vert_buf_pos: Subbuffer<[Vec2]>,
	pub buffer: Subbuffer<Mat4>,
	pub descriptor_set: Arc<PersistentDescriptorSet>,
	pub indirect_commands: Option<Subbuffer<[DrawIndirectCommand]>>,
}

#[derive(shipyard::Unique)]
pub struct Canvas
{
	base_dimensions: [u32; 2],
	projection: Mat4,
	scale_factor: f32,

	set_layout: Arc<DescriptorSetLayout>,
	text_set_layout: Arc<DescriptorSetLayout>,
	ui_pipeline: Arc<GraphicsPipeline>,
	text_pipeline: Arc<GraphicsPipeline>,

	gpu_resources: BTreeMap<EntityId, UiGpuResources>,

	quad_pos_buf: Subbuffer<[Vec2]>,
	quad_uv_buf: Subbuffer<[Vec2]>,

	default_font: Font<'static>,

	ui_cb: Option<Arc<SecondaryAutoCommandBuffer>>,
}
impl Canvas
{
	pub fn new(render_ctx: &mut RenderContext, canvas_width: u32, canvas_height: u32) -> Result<Self, GenericEngineError>
	{
		let device = render_ctx.descriptor_set_allocator().device().clone();

		let sampler_info = SamplerCreateInfo {
			mag_filter: Filter::Linear,
			min_filter: Filter::Linear,
			..Default::default()
		};
		let sampler = Sampler::new(device.clone(), sampler_info)?;
		let bindings = [
			DescriptorSetLayoutBinding {
				// binding 0: transformation matrix
				stages: ShaderStages::VERTEX,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
			},
			DescriptorSetLayoutBinding {
				// binding 1: tex
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![sampler],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
			},
		];
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: (0..).zip(bindings).collect(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

		let text_sampler = Sampler::new(device.clone(), SamplerCreateInfo::default())?;
		let text_bindings = [
			DescriptorSetLayoutBinding {
				// binding 0: transformation matrix
				stages: ShaderStages::VERTEX,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
			},
			DescriptorSetLayoutBinding {
				// binding 1: tex
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![text_sampler],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
			},
		];
		let text_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: (0..).zip(text_bindings).collect(),
			..Default::default()
		};
		let text_set_layout = DescriptorSetLayout::new(device.clone(), text_set_layout_info)?;

		let color_blend_state = ColorBlendState::with_attachment_states(
			1,
			ColorBlendAttachmentState {
				blend: Some(AttachmentBlend::alpha()),
				..Default::default()
			},
		);
		let rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
			..Default::default()
		};
		let ui_pipeline = crate::render::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleStrip,
			&[ui_vs::load(device.clone())?, ui_fs::load(device.clone())?],
			RasterizationState::default(),
			Some(color_blend_state.clone()),
			vec![set_layout.clone()],
			vec![],
			rendering_info.clone(),
			None,
		)?;
		let text_pipeline = crate::render::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleStrip,
			&[ui_text_vs::load(device.clone())?, ui_text_fs::load(device.clone())?],
			RasterizationState::default(),
			Some(color_blend_state),
			vec![text_set_layout.clone()],
			vec![],
			rendering_info,
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
		let default_font = Font::try_from_bytes(font_data as &[u8]).ok_or("Error constructing font")?;

		let dim = render_ctx.swapchain_dimensions();

		let (projection, scale_factor) = calculate_projection(canvas_width, canvas_height, dim[0], dim[1]);

		Ok(Canvas {
			base_dimensions: [canvas_width, canvas_height],
			projection,
			scale_factor,
			set_layout,
			text_set_layout,
			ui_pipeline,
			text_pipeline,
			gpu_resources: Default::default(),
			quad_pos_buf,
			quad_uv_buf,
			default_font,
			ui_cb: None,
		})
	}

	pub fn get_set_layout(&self) -> &Arc<DescriptorSetLayout>
	{
		&self.set_layout
	}

	pub fn get_pipeline(&self) -> &Arc<GraphicsPipeline>
	{
		&self.ui_pipeline
	}

	pub fn get_text_pipeline(&self) -> &Arc<GraphicsPipeline>
	{
		&self.text_pipeline
	}

	/// Run this function whenever the screen resizes, to adjust the canvas aspect ratio to fit.
	pub fn on_screen_resize(&mut self, screen_width: u32, screen_height: u32)
	{
		let (proj, scale_factor) =
			calculate_projection(self.base_dimensions[0], self.base_dimensions[1], screen_width, screen_height);
		self.projection = proj;
		self.scale_factor = scale_factor;
	}

	pub fn projection(&self) -> Mat4
	{
		self.projection
	}

	fn update_transform(
		&mut self,
		render_ctx: &mut RenderContext,
		eid: EntityId,
		transform: &super::UITransform,
		image_view: Arc<ImageView>,
		default_scale: Vec2,
		indirect_commands: Option<Subbuffer<[DrawIndirectCommand]>>,
		vert_buf_pos: Subbuffer<[Vec2]>,
	) -> Result<(), GenericEngineError>
	{
		let projected = self.projection
			* Mat4::from_scale_rotation_translation(
				transform.scale.unwrap_or(default_scale).extend(1.0),
				Quat::IDENTITY,
				transform.position.as_vec2().extend(0.0),
			);

		let buffer = match self.gpu_resources.get(&eid) {
			Some(resources) => {
				render_ctx.update_buffer(&[projected], resources.buffer.clone().into_slice())?;
				resources.buffer.clone()
			}
			None => render_ctx
				.new_buffer(&[projected], BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST)?
				.index(0),
		};

		let set_layout = if indirect_commands.is_some() {
			self.text_set_layout.clone()
		} else {
			self.set_layout.clone()
		};
		let writes = [
			WriteDescriptorSet::buffer(0, buffer.clone()),
			WriteDescriptorSet::image_view(1, image_view),
		];
		let descriptor_set = PersistentDescriptorSet::new(render_ctx.descriptor_set_allocator(), set_layout, writes, [])?;

		self.gpu_resources.insert(eid, UiGpuResources {
			vert_buf_pos,
			buffer,
			descriptor_set,
			indirect_commands,
		});

		Ok(())
	}

	/// Update the GPU resources for entities with a `Mesh` component.
	/// Call this whenever the component has been inserted or modified.
	pub fn update_mesh(
		&mut self,
		render_ctx: &mut RenderContext,
		eid: EntityId,
		transform: &super::UITransform,
		mesh: &super::mesh::Mesh,
	) -> Result<(), GenericEngineError>
	{
		if !mesh.image_path.as_os_str().is_empty() {
			let tex = render_ctx.get_texture(&mesh.image_path)?;
			let image_dimensions = UVec2::from(tex.dimensions()).as_vec2();
			self.update_transform(
				render_ctx,
				eid,
				transform,
				tex.view().clone(),
				image_dimensions,
				None,
				self.quad_pos_buf.clone(),
			)?;
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
	) -> Result<(), GenericEngineError>
	{
		let text_str = &text.text_str;

		// If the string is longer than the maximum image array layers allowed by Vulkan, refuse to
		// render it. On Windows/Linux desktop, the limit is always at least 2048, so it should be
		// extremely rare that we get such a long string, but we should check it anyways just in case.
		// We also check that it's no larger than 2048 characters, because we might use
		// vkCmdUpdateBuffer to update vertices, which has a limit of 65536 (32 * 2048) bytes.
		let image_format_info = ImageFormatInfo {
			format: Format::R8_UNORM,
			usage: ImageUsage::SAMPLED,
			..Default::default()
		};
		let max_glyphs: usize = render_ctx
			.device()
			.physical_device()
			.image_format_properties(image_format_info)?
			.unwrap()
			.max_array_layers
			.min(2048)
			.try_into()?;
		if text_str.chars().count() >= max_glyphs {
			log::warn!(
				"UI text string too long ({} chars, limit is {})! Refusing to render string: {}",
				text_str.len(),
				max_glyphs,
				text_str,
			);
			if let Some(resources) = self.gpu_resources.get_mut(&eid) {
				resources.indirect_commands = None;
			}
			return Ok(())
		}

		let glyph_image_array = text_to_image_array(text_str, &self.default_font, text.size * self.scale_factor);

		// If no visible glyphs were produced (e.g. the string was empty, or it only has space characters),
		// remove the indirect draw commands from the GPU resources, and then return immediately.
		if glyph_image_array.is_empty() {
			if let Some(resources) = self.gpu_resources.get_mut(&eid) {
				resources.indirect_commands = None;
			}
			return Ok(())
		}

		let img_dim = glyph_image_array
			.first()
			.map(|(image, _)| [image.width(), image.height()])
			.unwrap();

		let glyph_count = glyph_image_array.len();
		let prev_glyph_count = self
			.gpu_resources
			.get(&eid)
			.as_ref()
			.and_then(|res| res.indirect_commands.as_ref())
			.map(|commands| commands.len() as usize)
			.unwrap_or(0);

		let mut combined_images = Vec::with_capacity((img_dim[0] * img_dim[1]) as usize * glyph_count);
		let mut text_pos_verts = Vec::with_capacity(glyph_count * 4);
		for (image, offset) in glyph_image_array.into_iter() {
			combined_images.extend_from_slice(image.into_raw().as_slice());

			let first_corner: Vec2 = offset / self.scale_factor;
			let img_dim_scaled = UVec2::from(img_dim).as_vec2() / self.scale_factor;
			text_pos_verts.extend_from_slice(&[
				first_corner,
				first_corner + Vec2::new(0.0, img_dim_scaled.y),
				first_corner + Vec2::new(img_dim_scaled.x, 0.0),
				first_corner + img_dim_scaled,
			]);
		}

		let tex = render_ctx.new_texture_from_slice(&combined_images, Format::R8_UNORM, img_dim, 1, glyph_count.try_into()?)?;

		let (indirect_commands, vert_buf_pos);
		if glyph_count == prev_glyph_count {
			// Reuse buffers if they're of the same length.
			// If `prev_glyph_count` is greater than 0, we already know that `gpu_resources` for
			// the given `eid` is `Some`, so we use `unwrap` here.
			let resources = self.gpu_resources.get(&eid).unwrap();

			indirect_commands = resources.indirect_commands.clone().unwrap();

			render_ctx.update_buffer(&text_pos_verts, resources.vert_buf_pos.clone())?;
			vert_buf_pos = resources.vert_buf_pos.clone();
		} else {
			let mut commands = Vec::with_capacity(glyph_count);
			for i in 0..glyph_count {
				commands.push(DrawIndirectCommand {
					vertex_count: 4,
					instance_count: 1,
					first_vertex: (i * 4) as u32,
					first_instance: 0,
				});
			}
			indirect_commands = render_ctx.new_buffer(&commands, BufferUsage::INDIRECT_BUFFER)?;

			let vbo_usage = BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST;
			vert_buf_pos = render_ctx.new_buffer(&text_pos_verts, vbo_usage)?;
		}

		self.update_transform(
			render_ctx,
			eid,
			transform,
			tex.view().clone(),
			Vec2::ONE,
			Some(indirect_commands),
			vert_buf_pos,
		)?;

		Ok(())
	}

	pub fn draw_text(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		eid: EntityId,
	) -> Result<(), GenericEngineError>
	{
		if let Some(resources) = self.gpu_resources.get(&eid) {
			if let Some(indirect_commands) = resources.indirect_commands.clone() {
				cb.bind_descriptor_sets(
					vulkano::pipeline::PipelineBindPoint::Graphics,
					self.text_pipeline.layout().clone(),
					0,
					vec![resources.descriptor_set.clone()],
				)?;
				cb.bind_vertex_buffers(0, (resources.vert_buf_pos.clone(),))?;
				cb.draw_indirect(indirect_commands)?;
			}
		}
		Ok(())
	}

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		eid: EntityId,
		component: &super::mesh::Mesh
	) -> Result<(), GenericEngineError>
	{
		if let Some(resources) = self.gpu_resources.get(&eid) {
			cb.bind_descriptor_sets(
				vulkano::pipeline::PipelineBindPoint::Graphics,
				self.ui_pipeline.layout().clone(),
				0,
				vec![resources.descriptor_set.clone()],
			)?;

			match component.mesh_type {
				MeshType::Quad => {
					cb.bind_vertex_buffers(0, (self.quad_pos_buf.clone(), self.quad_uv_buf.clone()))?;
					cb.draw(4, 1, 0, 0)?;
				}
				MeshType::Frame(_border_width) => {
					todo!();
				}
			}
		}

		Ok(())
	}

	pub fn add_cb(&mut self, command_buffer: Arc<SecondaryAutoCommandBuffer>)
	{
		self.ui_cb = Some(command_buffer);
	}
	pub fn take_cb(&mut self) -> Option<Arc<SecondaryAutoCommandBuffer>>
	{
		self.ui_cb.take()
	}
}

fn calculate_projection(canvas_width: u32, canvas_height: u32, screen_width: u32, screen_height: u32) -> (Mat4, f32)
{
	let canvas_aspect_ratio = canvas_width as f32 / canvas_height as f32;
	let screen_aspect_ratio = screen_width as f32 / screen_height as f32;

	// UI scale factor, used to increase resolution of components such as text when necessary
	let scale_factor;

	// Adjusted canvas dimensions
	let (adjusted_canvas_w, adjusted_canvas_h);

	// If the screen is wider than the canvas, make the canvas wider.
	// Otherwise, make the canvas taller.
	if screen_aspect_ratio > canvas_aspect_ratio {
		adjusted_canvas_w = canvas_height * screen_width / screen_height;
		adjusted_canvas_h = canvas_height;
		scale_factor = screen_height as f32 / canvas_height as f32;
	} else {
		adjusted_canvas_w = canvas_width;
		adjusted_canvas_h = canvas_width * screen_height / screen_width;
		scale_factor = screen_width as f32 / canvas_width as f32;
	}

	let half_width = adjusted_canvas_w as f32 / 2.0;
	let half_height = adjusted_canvas_h as f32 / 2.0;
	let proj = Mat4::orthographic_lh(-half_width, half_width, -half_height, half_height, 0.0, 1.0);

	(proj, scale_factor)
}

/// Create a `Vec` of greyscale images.
/// Each image is paired with a `Vec2` representing the top left corner of each image relative to the baseline.
fn text_to_image_array(text: &str, font: &Font<'static>, size: f32) -> Vec<(GrayImage, Vec2)>
{
	let scale_uniform = Scale::uniform(size);
	let glyphs: Vec<_> = font.layout(text, scale_uniform, rusttype::point(0.0, 0.0)).collect();

	// Get the largest glyphs in terms of width and height respectively
	let max_width: u32 = glyphs
		.iter()
		.filter_map(|glyph| glyph.pixel_bounding_box())
		.map(|bb| bb.width().abs() as u32)
		.max()
		.unwrap_or(1);
	let max_height: u32 = glyphs
		.iter()
		.filter_map(|glyph| glyph.pixel_bounding_box())
		.map(|bb| bb.height().abs() as u32)
		.max()
		.unwrap_or(1);

	let mut bitmaps = Vec::with_capacity(glyphs.len());
	for glyph in glyphs {
		if let Some(bb) = glyph.pixel_bounding_box() {
			let mut image = DynamicImage::new_luma8(max_width, max_height).into_luma8();

			// Draw the glyph into the image per-pixel by using the draw closure,
			// and turn the coverage into an alpha value
			glyph.draw(|x, y, v| image.put_pixel(x, y, Luma([(v * 255.0) as u8])));

			let pos = Vec2::new(bb.min.x as f32, bb.min.y as f32);

			bitmaps.push((image, pos));
		}
	}

	bitmaps
}
