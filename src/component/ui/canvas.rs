/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use image::{DynamicImage, RgbaImage, Rgba};
use rusttype::{point, Font, Scale};
use shipyard::EntityId;
use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo}, view::ImageView};
use vulkano::pipeline::{
	graphics::{color_blend::AttachmentBlend, input_assembly::PrimitiveTopology, GraphicsPipeline},
	Pipeline,
};
use vulkano::shader::ShaderStages;

use crate::render::RenderContext;
use crate::GenericEngineError;
use super::mesh::MeshType;

mod ui_vs {
	vulkano_shaders::shader! {
		ty: "vertex",
		bytes: "shaders/ui.vert.spv",
	}
}
mod ui_fs {
	vulkano_shaders::shader! {
		ty: "fragment",
		bytes: "shaders/ui.frag.spv",
	}
}

#[derive(shipyard::Unique)]
pub struct Canvas
{
	base_dimensions: [u32; 2],
	projection: Mat4,

	set_layout: Arc<DescriptorSetLayout>,
	ui_pipeline: Arc<GraphicsPipeline>,

	gpu_resources: BTreeMap<EntityId, (super::mesh::MeshType, Arc<PersistentDescriptorSet>)>,

	quad_pos_buf: Subbuffer<[Vec2]>,
	quad_uv_buf: Subbuffer<[Vec2]>,

	default_font: Font<'static>,
}
impl Canvas
{
	pub fn new(render_ctx: &mut RenderContext, canvas_width: u32, canvas_height: u32, screen_width: u32, screen_height: u32)
		-> Result<Self, GenericEngineError>
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
				(0, DescriptorSetLayoutBinding { // binding 0: transformation matrix
					stages: ShaderStages::VERTEX,
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
				}),
				(1, DescriptorSetLayoutBinding { // binding 1: sampler0
					stages: ShaderStages::FRAGMENT,
					immutable_samplers: vec![ sampler ],
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
				}),
				(2, DescriptorSetLayoutBinding { // binding 2: texture
					stages: ShaderStages::FRAGMENT,
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
				}),
			].into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

		let ui_pipeline_config = crate::render::pipeline::PipelineConfig {
			vertex_shader: ui_vs::load(device.clone())?,
			fragment_shader: ui_fs::load(device.clone())?,
			fragment_shader_transparency: None,
			attachment_blend: Some(AttachmentBlend::alpha()),
			primitive_topology: PrimitiveTopology::TriangleStrip,
			depth_processing: false,
			set_layouts: vec![ set_layout.clone() ],
			push_constant_ranges: vec![],
		};
		let ui_pipeline = crate::render::pipeline::new_from_config(device, ui_pipeline_config)?;

		let vbo_usage = BufferUsage::VERTEX_BUFFER;
		let quad_pos_verts = [
			Vec2::new(-0.5, -0.5),
			Vec2::new(-0.5, 0.5),
			Vec2::new(0.5, -0.5),
			Vec2::new(0.5, 0.5),
		];
		let quad_pos_buf = render_ctx.new_buffer_from_iter(quad_pos_verts, vbo_usage)?;
		let quad_uv_verts = [
			Vec2::new(0.0, 0.0),
			Vec2::new(0.0, 1.0),
			Vec2::new(1.0, 0.0),
			Vec2::new(1.0, 1.0),
		];
		let quad_uv_buf = render_ctx.new_buffer_from_iter(quad_uv_verts, vbo_usage)?;

		let font_data = include_bytes!("../../../resource/mplus-1m-medium.ttf");
		let default_font = Font::try_from_bytes(font_data as &[u8]).ok_or("Error constructing font")?;

		Ok(Canvas {
			base_dimensions: [canvas_width, canvas_height],
			projection: calculate_projection(canvas_width, canvas_height, screen_width, screen_height),
			set_layout,
			ui_pipeline,
			gpu_resources: Default::default(),
			quad_pos_buf,
			quad_uv_buf,
			default_font,
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
		transform: &super::UITransform,
		image_view: Arc<ImageView>,
		image_dimensions: Vec2,
	) -> Result<Arc<PersistentDescriptorSet>, GenericEngineError>
	{
		let projected = self.projection * Mat4::from_scale_rotation_translation(
			transform.scale.unwrap_or(image_dimensions).extend(0.0), 
			Quat::IDENTITY, 
			transform.position.as_vec2().extend(0.0)
		);
		let buf = render_ctx.new_buffer_from_data(projected, BufferUsage::UNIFORM_BUFFER)?;
		let set = PersistentDescriptorSet::new(
			render_ctx.descriptor_set_allocator(),
			self.set_layout.clone(),
			[
				WriteDescriptorSet::buffer(0, buf),
				WriteDescriptorSet::image_view(2, image_view),
			],
			[],
		)?;

		Ok(set)
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

			let set = self.update_transform(render_ctx, transform, tex.view(), image_dimensions)?;

			self.gpu_resources.insert(eid, (mesh.mesh_type, set));
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
		let text_image = text_to_image(&text.text_str, &self.default_font, text.size)?;
		let img_dim = [text_image.width(), text_image.height()];
		let tex = render_ctx.new_texture_from_iter(text_image.into_raw(), Format::R8G8B8A8_SRGB, img_dim, 1)?;

		let img_dim_vec2 = Vec2::new(img_dim[0] as f32, img_dim[1] as f32);

		let set = self.update_transform(render_ctx, transform, tex.view(), img_dim_vec2)?;

		self.gpu_resources.insert(eid, (super::mesh::MeshType::Quad, set));

		Ok(())
	}

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		eid: EntityId,
	) -> Result<(), GenericEngineError>
	{
		if let Some((mesh_type, descriptor_set)) = self.gpu_resources.get(&eid) {
			cb.bind_descriptor_sets(
				vulkano::pipeline::PipelineBindPoint::Graphics, 
				self.ui_pipeline.layout().clone(),
				0,
				vec![ descriptor_set.clone() ]
			)?;

			match mesh_type {
				MeshType::Quad => {
					cb.bind_vertex_buffers(0, (self.quad_pos_buf.clone(), self.quad_uv_buf.clone()))?;
					cb.draw(4, 1, 0, 0)?;
				}
				MeshType::Frame(_border_width) => {
					// TODO: implement
				}
			}
		}

		Ok(())
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

fn text_to_image(text: &str, font: &Font<'static>, size: f32) -> Result<RgbaImage, GenericEngineError>
{
	let scale_uniform = Scale::uniform(size);
	let color = (255, 255, 0);
	let v_metrics = font.v_metrics(scale_uniform);

	// lay out the glyphs in a line with 1 pixel padding
	let glyphs: Vec<_> = font
		.layout(text, scale_uniform, point(1.0, 1.0 + v_metrics.ascent))
		.collect();

	// work out the layout size
	let glyphs_height = (v_metrics.ascent - v_metrics.descent).ceil() as u32;
	let min_x = glyphs
		.first()
		.ok_or("there were no glyphs for the string!")?
		.pixel_bounding_box()
		.ok_or("pixel_bounding_box was `None`!")?
		.min
		.x;
	let max_x = glyphs
		.last()
		.ok_or("there were no glyphs for the string!")?
		.pixel_bounding_box()
		.ok_or("pixel_bounding_box was `None`!")?
		.max
		.x;
	let glyphs_width = (max_x - min_x) as u32;

	// Create a new rgba image
	let mut image = DynamicImage::new_rgba8(glyphs_width + 2, glyphs_height + 2).into_rgba8();

	// Loop through the glyphs in the text, positing each one on a line
	for glyph in glyphs {
		if let Some(bounding_box) = glyph.pixel_bounding_box() {
			// Draw the glyph into the image per-pixel by using the draw closure
			glyph.draw(|x, y, v| {
				// Offset the position by the glyph bounding box
				let x_offset = x + bounding_box.min.x as u32;
				let y_offset = y + bounding_box.min.y as u32;
				// Make sure the pixel isn't out of bounds. If it is OoB, then don't draw it.
				if x_offset >= image.width() || y_offset >= image.height() {
					log::warn!(
						"Text pixel at ({},{}) is out of bounds of ({},{})",
						x_offset,
						y_offset,
						image.width(),
						image.height()
					);
				} else {
					// Turn the coverage into an alpha value
					image.put_pixel(x_offset, y_offset, Rgba([color.0, color.1, color.2, (v * 255.0) as u8]))
				}
			});
		}
	}

	// TODO: use these to properly align the image
	let _mesh_top_left = Vec2::new(image.width() as f32 / -2.0, -v_metrics.ascent - 1.0);
	let _mesh_bottom_right = Vec2::new(image.height() as f32 / 2.0, -v_metrics.descent + 1.0);

	Ok(image)
}
