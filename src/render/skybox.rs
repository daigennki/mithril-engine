/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use std::path::PathBuf;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo, SubpassContents,
};
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{
	sampler::{Sampler, SamplerCreateInfo},
	view::{ImageView, ImageViewCreateInfo, ImageViewType},
	ImageCreateFlags, ImageCreateInfo, ImageUsage,
};
use vulkano::pipeline::graphics::{
	color_blend::ColorBlendState, input_assembly::PrimitiveTopology, rasterization::RasterizationState,
	subpass::PipelineRenderingCreateInfo, GraphicsPipeline,
};
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{graphics::viewport::Viewport, Pipeline, PipelineBindPoint, PipelineLayout};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

use super::RenderContext;

mod vs
{
	vulkano_shaders::shader! {
		ty: "vertex",
		src: r"
			#version 450

			layout(push_constant) uniform pc
			{
				mat4 sky_projview;
			};
			
			layout(location = 0) in vec3 position;
			layout(location = 0) out vec3 cube_pos; // give original vertex position to fragment shader

			void main()
			{
				cube_pos = position;
				vec4 new_pos = sky_projview * vec4(position, 1.0);
				gl_Position = new_pos.xyww;
			}
		",
	}
}
mod fs
{
	vulkano_shaders::shader! {
		ty: "fragment",
		src: r"
			#version 450

			layout(binding = 0) uniform samplerCube sky_tex;

			layout(location = 0) in vec3 cube_pos;
			layout(location = 0) out vec4 color_out;

			void main()
			{
				color_out = texture(sky_tex, cube_pos.xzy);
			}
		",
	}
}

#[derive(Copy, Clone, bytemuck::AnyBitPattern)]
#[repr(C)]
struct SkyCubeData
{
	position: [f32; 24],
	indices: [u16; 17],
}
const SKY_CUBE_DATA: SkyCubeData = SkyCubeData {
	// Sky cube, consisting of two fans with the "center" being opposite corners of the cube.
	// Relative to camera at default state, -X is left, +Y is forward, and +Z is up.
	#[rustfmt::skip]
	position: [
		-1.0, -1.0, -1.0,
		-1.0, -1.0, 1.0,
		1.0, -1.0, 1.0,
		1.0, -1.0, -1.0,
		1.0, 1.0, -1.0,
		-1.0, 1.0, -1.0,
		-1.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
	],

	#[rustfmt::skip]
	indices: [
		0, 1, 2, 3, 4, 5, 6, 1, u16::MAX,
		7, 1, 2, 3, 4, 5, 6, 1,
	],
};

#[derive(shipyard::Unique)]
pub struct Skybox
{
	sky_pipeline: Arc<GraphicsPipeline>,
	cube_vbo: Subbuffer<[f32]>,
	cube_ibo: Subbuffer<[u16]>,
	descriptor_set: Arc<PersistentDescriptorSet>,
}
impl Skybox
{
	/// Create a new skybox, using 6 texture files for each face, loaded from paths in the given format `tex_files_format`.
	/// The format string should have an asterisk in it, for example "sky/Daylight Box_*.png", which will be replaced
	/// with the face name.
	/// Face names are "Right", "Left", "Top", "Bottom", "Front", and "Back".
	pub fn new(render_ctx: &mut RenderContext, tex_files_format: String) -> crate::Result<Self>
	{
		let device = render_ctx.descriptor_set_allocator.device().clone();

		let cubemap_sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear_no_mipmap())?;
		let tex_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			immutable_samplers: vec![cubemap_sampler],
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
		};
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [(0, tex_binding)].into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

		let pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![set_layout.clone()],
			push_constant_ranges: vec![PushConstantRange {
				stages: ShaderStages::VERTEX,
				offset: 0,
				size: std::mem::size_of::<Mat4>().try_into().unwrap(),
			}],
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_info)?;

		let rendering_formats = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
			..Default::default()
		};
		let sky_pipeline = super::new_graphics_pipeline(
			PrimitiveTopology::TriangleFan,
			&[vs::load(device.clone())?, fs::load(device.clone())?],
			RasterizationState::default(),
			pipeline_layout,
			rendering_formats,
			Some(ColorBlendState::with_attachment_states(1, Default::default())),
			None,
		)?;

		// sky texture cubemap
		let face_names = ["Right", "Left", "Top", "Bottom", "Front", "Back"];
		let face_paths = face_names.map(|face_name| tex_files_format.replace('*', face_name).into());
		let sky_cubemap = new_cubemap(render_ctx, face_paths)?;
		let descriptor_set = PersistentDescriptorSet::new(
			&render_ctx.descriptor_set_allocator,
			set_layout,
			[WriteDescriptorSet::image_view(0, sky_cubemap)],
			[],
		)?;

		let cube_buffer = render_ctx.new_buffer(vec![SKY_CUBE_DATA], BufferUsage::VERTEX_BUFFER | BufferUsage::INDEX_BUFFER)?;
		let (cube_vbo_bytes, cube_ibo_bytes) = cube_buffer.into_bytes().split_at(std::mem::size_of::<[f32; 24]>() as u64);

		Ok(Skybox {
			sky_pipeline,
			cube_vbo: cube_vbo_bytes.reinterpret(),
			cube_ibo: cube_ibo_bytes.reinterpret(),
			descriptor_set,
		})
	}

	pub fn draw(
		&mut self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		color_image: Arc<ImageView>,
		sky_projview: Mat4,
	) -> crate::Result<()>
	{
		let viewport_extent = color_image.image().extent();
		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [viewport_extent[0] as f32, viewport_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};

		let sky_render_info = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::DontCare,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(color_image)
			})],
			contents: SubpassContents::Inline,
			..Default::default()
		};

		cb_builder
			.begin_rendering(sky_render_info)?
			.set_viewport(0, [viewport].as_slice().into())?
			.bind_pipeline_graphics(self.sky_pipeline.clone())?
			.bind_descriptor_sets(
				PipelineBindPoint::Graphics,
				self.sky_pipeline.layout().clone(),
				0,
				vec![self.descriptor_set.clone()],
			)?
			.push_constants(self.sky_pipeline.layout().clone(), 0, sky_projview)?
			.bind_vertex_buffers(0, vec![self.cube_vbo.clone()])?
			.bind_index_buffer(self.cube_ibo.clone())?
			.draw_indexed(17, 1, 0, 0, 0)?
			.end_rendering()?;

		Ok(())
	}
}

/// Load six image files as cubemap textures into memory.
///
/// `faces` is paths to textures of each face of the cubemap, in order of +X, -X, +Y, -Y, +Z, -Z.
///
/// Unlike `new_texture`, the results of this are *not* cached.
fn new_cubemap(render_ctx: &mut RenderContext, faces: [PathBuf; 6]) -> crate::Result<Arc<ImageView>>
{
	let mut combined_data = Vec::<u8>::new();
	let mut cube_fmt = None;
	let mut cube_dim = None;

	for face_path in faces {
		let (face_fmt, face_dim, _, img_raw) = super::load_texture(&face_path)?;

		if face_fmt != *cube_fmt.get_or_insert(face_fmt) {
			return Err("Not all faces of a cubemap have the same format!".into());
		}
		if face_dim != *cube_dim.get_or_insert(face_dim) {
			return Err("Not all faces of a cubemap have the same dimensions!".into());
		}

		let mip_size = super::get_mip_size(face_fmt, face_dim[0], face_dim[1]).try_into().unwrap();
		if combined_data.capacity() == 0 {
			combined_data.reserve(mip_size * 6);
		}
		combined_data.extend(&img_raw[..mip_size]);
	}

	let extent = cube_dim.unwrap();
	let image_info = ImageCreateInfo {
		flags: ImageCreateFlags::CUBE_COMPATIBLE,
		format: cube_fmt.unwrap(),
		extent: [extent[0], extent[1], 1],
		array_layers: 6,
		usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
		..Default::default()
	};
	let image = render_ctx.new_image(combined_data, image_info)?;

	let view_create_info = ImageViewCreateInfo {
		view_type: ImageViewType::Cube,
		..ImageViewCreateInfo::from_image(&image)
	};
	Ok(ImageView::new(image, view_create_info)?)
}
