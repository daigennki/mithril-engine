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
	sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
	view::ImageView,
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
			layout(location = 1) flat out int draw_id;

			void main()
			{
				gl_Position = transformation * vec4(pos, 0.0, 1.0);
				texcoord = uv;
				draw_id = gl_DrawID;
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
			layout(location = 1) flat in int draw_id;
			layout(location = 0) out vec4 color_out;

			void main()
			{
				color_out = texture(tex, texcoord);
				color_out.rgb *= color_out.a;
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
	pub mesh_type: super::mesh::MeshType,
	pub vert_buf_pos: Subbuffer<[Vec2]>,
	pub vert_buf_uv: Subbuffer<[Vec2]>,
	pub buffer: Subbuffer<Mat4>,
	pub descriptor_set: Arc<PersistentDescriptorSet>,
	pub indirect_commands: Option<Subbuffer<[DrawIndirectCommand]>>,
}

#[derive(shipyard::Unique)]
pub struct Canvas
{
	base_dimensions: [u32; 2],
	projection: Mat4,

	set_layout: Arc<DescriptorSetLayout>,
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
			address_mode: [
				SamplerAddressMode::ClampToEdge,
				SamplerAddressMode::ClampToEdge,
				SamplerAddressMode::ClampToEdge,
			],
			..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
		};
		let sampler = Sampler::new(device.clone(), sampler_info)?;

		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(
					0,
					DescriptorSetLayoutBinding {
						// binding 0: transformation matrix
						stages: ShaderStages::VERTEX,
						..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
					},
				),
				(
					1,
					DescriptorSetLayoutBinding {
						// binding 1: tex
						stages: ShaderStages::FRAGMENT,
						immutable_samplers: vec![sampler],
						..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
					},
				),
			]
			.into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

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
			&[ui_vs::load(device.clone())?, ui_text_fs::load(device.clone())?],
			RasterizationState::default(),
			Some(color_blend_state),
			vec![set_layout.clone()],
			vec![],
			rendering_info,
			None,
		)?;

		let vbo_usage = BufferUsage::VERTEX_BUFFER;
		let quad_pos_verts = [
			Vec2::new(-0.5, -0.5),
			Vec2::new(-0.5, 0.5),
			Vec2::new(0.5, -0.5),
			Vec2::new(0.5, 0.5),
		];
		let quad_pos_buf = render_ctx.new_buffer(&quad_pos_verts, vbo_usage)?;
		let quad_uv_verts = [
			Vec2::new(0.0, 0.0),
			Vec2::new(0.0, 1.0),
			Vec2::new(1.0, 0.0),
			Vec2::new(1.0, 1.0),
		];
		let quad_uv_buf = render_ctx.new_buffer(&quad_uv_verts, vbo_usage)?;

		let font_data = include_bytes!("../../../resource/mplus-1m-medium.ttf");
		let default_font = Font::try_from_bytes(font_data as &[u8]).ok_or("Error constructing font")?;

		let dim = render_ctx.swapchain_dimensions();

		Ok(Canvas {
			base_dimensions: [canvas_width, canvas_height],
			projection: calculate_projection(canvas_width, canvas_height, dim[0], dim[1]),
			set_layout,
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
		self.projection = calculate_projection(self.base_dimensions[0], self.base_dimensions[1], screen_width, screen_height)
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
		image_dimensions: Vec2,
		mesh_type: super::mesh::MeshType,
		glyph_offsets: Option<Vec<Vec2>>,
	) -> Result<(), GenericEngineError>
	{
		let projected = if glyph_offsets.is_some() {
			// for text components
			self.projection
				* Mat4::from_scale_rotation_translation(
					Vec3::ONE,
					Quat::IDENTITY,
					transform.position.as_vec2().extend(0.0),
				)
		} else {
			// for quad/mesh components
			self.projection
				* Mat4::from_scale_rotation_translation(
					transform.scale.unwrap_or(image_dimensions).extend(0.0),
					Quat::IDENTITY,
					transform.position.as_vec2().extend(0.0),
				)
		};

		let buffer = match self.gpu_resources.get(&eid) {
			Some(resources) => {
				render_ctx.update_buffer(&[projected], resources.buffer.clone().into_slice())?;
				resources.buffer.clone()
			}
			None => render_ctx
				.new_buffer(&[projected], BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST)?
				.index(0),
		};

		let descriptor_set = PersistentDescriptorSet::new(
			render_ctx.descriptor_set_allocator(),
			self.set_layout.clone(),
			[
				WriteDescriptorSet::buffer(0, buffer.clone()),
				WriteDescriptorSet::image_view(1, image_view),
			],
			[],
		)?;

		let new_glyph_count = glyph_offsets.as_ref().map(|offsets| offsets.len()).unwrap_or(0);
		let (vert_buf_pos, vert_buf_uv, indirect_commands, prev_glyph_count);
		if let Some(offsets) = glyph_offsets {
			// reuse buffers when they're of the same length
			match self.gpu_resources.get(&eid) {
				Some(resources) => {
					prev_glyph_count = resources
						.indirect_commands
						.as_ref()
						.map(|commands| commands.len())
						.unwrap_or(0)
						.try_into()?;

					if new_glyph_count == prev_glyph_count {
						indirect_commands = resources.indirect_commands.clone();
					} else {
						let mut commands = Vec::with_capacity(offsets.len().try_into()?);
						for i in 0..offsets.len() {
							let command = DrawIndirectCommand {
								vertex_count: 4,
								instance_count: 1,
								first_vertex: (i * 4).try_into()?,
								first_instance: 0,
							};
							commands.push(command);
						}
						indirect_commands = Some(render_ctx.new_buffer(commands.as_slice(), BufferUsage::INDIRECT_BUFFER)?);
					}
				}
				None => {
					prev_glyph_count = 0;
					let mut commands = Vec::with_capacity(offsets.len().try_into()?);
					for i in 0..offsets.len() {
						let command = DrawIndirectCommand {
							vertex_count: 4,
							instance_count: 1,
							first_vertex: (i * 4).try_into()?,
							first_instance: 0,
						};
						commands.push(command);
					}
					indirect_commands = Some(render_ctx.new_buffer(commands.as_slice(), BufferUsage::INDIRECT_BUFFER)?);
				}
			}
			
			let mut text_pos_verts = Vec::with_capacity(4 * offsets.len());
			let mut text_uv_verts = Vec::with_capacity(4 * offsets.len());
			for offset in offsets {
				let first_corner: Vec2 = offset.into();

				text_pos_verts.extend_from_slice(&[
					first_corner,
					first_corner + Vec2::new(0.0, image_dimensions[1]),
					first_corner + Vec2::new(image_dimensions[0], 0.0),
					first_corner + image_dimensions,
				]);

				text_uv_verts.extend_from_slice(&[
					Vec2::new(0.0, 0.0),
					Vec2::new(0.0, 1.0),
					Vec2::new(1.0, 0.0),
					Vec2::new(1.0, 1.0),
				]);
			}
			
			let vbo_usage = BufferUsage::VERTEX_BUFFER;
			match self.gpu_resources.get(&eid) {
				Some(resources) => {
					if new_glyph_count == prev_glyph_count {
						// reuse buffers when they're of the same length
						render_ctx.update_buffer(&text_pos_verts, resources.vert_buf_pos.clone())?;
						vert_buf_pos = resources.vert_buf_pos.clone();
						vert_buf_uv = resources.vert_buf_uv.clone();
					} else {
						vert_buf_pos = render_ctx.new_buffer(&text_pos_verts, vbo_usage | BufferUsage::TRANSFER_DST)?;
						vert_buf_uv = render_ctx.new_buffer(&text_uv_verts, vbo_usage)?;
					}
				}
				None => {
					vert_buf_pos = render_ctx.new_buffer(&text_pos_verts, vbo_usage | BufferUsage::TRANSFER_DST)?;
					vert_buf_uv = render_ctx.new_buffer(&text_uv_verts, vbo_usage)?;
				}
			};
		} else {
			vert_buf_pos = self.quad_pos_buf.clone();
			vert_buf_uv = self.quad_uv_buf.clone();
			indirect_commands = None;
		}

		self.gpu_resources.insert(eid, UiGpuResources {
			vert_buf_pos,
			vert_buf_uv,
			buffer,
			descriptor_set,
			mesh_type,
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
		// Only actually do something if the `Mesh` is supposed to use an image file (path is not empty),
		// rather than a texture set by another component like `UIText`.
		if !mesh.image_path.as_os_str().is_empty() {
			let tex = render_ctx.get_texture(&mesh.image_path)?;
			let tex_dimensions = tex.dimensions();
			let image_dimensions = Vec2::new(tex_dimensions[0] as f32, tex_dimensions[1] as f32);

			self.update_transform(
				render_ctx,
				eid,
				transform,
				tex.view().clone(),
				image_dimensions,
				mesh.mesh_type,
				None,
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
		let glyph_image_array = text_to_image_array(&text.text_str, &self.default_font, text.size);

		let img_dim = glyph_image_array
			.first()
			.map(|(image, _)| [image.width(), image.height()])
			.unwrap();

		let mut combined_images = Vec::with_capacity((img_dim[0] * img_dim[1]) as usize * glyph_image_array.len());
		let mut glyph_offsets = Vec::with_capacity(glyph_image_array.len());
		for (image, offset) in glyph_image_array.into_iter() {
			combined_images.extend_from_slice(image.into_raw().as_slice());
			glyph_offsets.push(offset);
		}
		let array_layers = glyph_offsets.len().try_into()?;
		let tex = render_ctx.new_texture_from_slice(&combined_images, Format::R8_UNORM, img_dim, 1, array_layers)?;

		let img_dim_vec2 = Vec2::new(img_dim[0] as f32, img_dim[1] as f32);

		self.update_transform(
			render_ctx,
			eid,
			transform,
			tex.view().clone(),
			img_dim_vec2,
			super::mesh::MeshType::Quad,
			Some(glyph_offsets),
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
			cb.bind_descriptor_sets(
				vulkano::pipeline::PipelineBindPoint::Graphics,
				self.text_pipeline.layout().clone(),
				0,
				vec![resources.descriptor_set.clone()],
			)?;
			cb.bind_vertex_buffers(0, (resources.vert_buf_pos.clone(), resources.vert_buf_uv.clone()))?;
			cb.draw_indirect(resources.indirect_commands.clone().unwrap())?;
		}
		Ok(())
	}

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		eid: EntityId,
	) -> Result<(), GenericEngineError>
	{
		if let Some(resources) = self.gpu_resources.get(&eid) {
			cb.bind_descriptor_sets(
				vulkano::pipeline::PipelineBindPoint::Graphics,
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

fn calculate_projection(canvas_width: u32, canvas_height: u32, screen_width: u32, screen_height: u32) -> Mat4
{
	let canvas_aspect_ratio = canvas_width as f32 / canvas_height as f32;
	let screen_aspect_ratio = screen_width as f32 / screen_height as f32;

	// adjusted canvas dimensions
	let mut adj_canvas_w = canvas_width;
	let mut adj_canvas_h = canvas_height;

	// if the screen is wider than the canvas, make the canvas wider.
	// otherwise, make the canvas taller.
	if screen_aspect_ratio > canvas_aspect_ratio {
		adj_canvas_w = canvas_height * screen_width / screen_height;
	} else {
		adj_canvas_h = canvas_width * screen_height / screen_width;
	}

	let half_width = adj_canvas_w as f32 / 2.0;
	let half_height = adj_canvas_h as f32 / 2.0;
	Mat4::orthographic_lh(-half_width, half_width, -half_height, half_height, 0.0, 1.0)
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
