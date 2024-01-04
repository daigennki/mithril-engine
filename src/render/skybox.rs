/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
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
	sampler::{Sampler, SamplerCreateInfo},
	view::ImageView,
};
use vulkano::pipeline::graphics::{
	color_blend::ColorBlendState, input_assembly::PrimitiveTopology, rasterization::RasterizationState,
	subpass::PipelineRenderingCreateInfo, GraphicsPipeline,
};
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{Pipeline, PipelineBindPoint, PipelineLayout};
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
	command_buffer: Option<Arc<SecondaryAutoCommandBuffer>>,
}
impl Skybox
{
	/// Create a new skybox, using 6 texture files for each face, loaded from paths in the given format `tex_files_format`.
	/// The format string should have an asterisk in it, for example "sky/Daylight Box_*.png", which will be replaced
	/// with the face name.
	/// Face names are "Right", "Left", "Top", "Bottom", "Front", and "Back".
	pub fn new(render_ctx: &mut RenderContext, tex_files_format: String) -> crate::Result<Self>
	{
		let device = render_ctx.descriptor_set_allocator().device().clone();

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
		let sky_pipeline = super::pipeline::new(
			device.clone(),
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
		let sky_cubemap = render_ctx.new_cubemap_texture(face_paths)?;
		let descriptor_set = PersistentDescriptorSet::new(
			render_ctx.descriptor_set_allocator(),
			set_layout,
			[WriteDescriptorSet::image_view(0, sky_cubemap.view().clone())],
			[],
		)?;

		let cube_buffer = render_ctx.new_buffer(&[SKY_CUBE_DATA], BufferUsage::VERTEX_BUFFER | BufferUsage::INDEX_BUFFER)?;
		let (cube_vbo_bytes, cube_ibo_bytes) = cube_buffer.into_bytes().split_at(std::mem::size_of::<[f32; 24]>() as u64);

		Ok(Skybox {
			sky_pipeline,
			cube_vbo: cube_vbo_bytes.reinterpret(),
			cube_ibo: cube_ibo_bytes.reinterpret(),
			descriptor_set,
			command_buffer: None,
		})
	}

	pub fn draw(&mut self, render_ctx: &RenderContext, sky_projview: Mat4) -> crate::Result<()>
	{
		let vp_extent = render_ctx.swapchain_dimensions();
		let mut cb = render_ctx.gather_commands(&[Format::R16G16B16A16_SFLOAT], None, None, vp_extent)?;

		cb.bind_pipeline_graphics(self.sky_pipeline.clone())?
			.bind_descriptor_sets(
				PipelineBindPoint::Graphics,
				self.sky_pipeline.layout().clone(),
				0,
				vec![self.descriptor_set.clone()],
			)?
			.push_constants(self.sky_pipeline.layout().clone(), 0, sky_projview)?
			.bind_vertex_buffers(0, vec![self.cube_vbo.clone()])?
			.bind_index_buffer(self.cube_ibo.clone())?
			.draw_indexed(17, 1, 0, 0, 0)?;

		assert!(self.command_buffer.replace(cb.build()?).is_none());

		Ok(())
	}

	pub fn execute_rendering(
		&mut self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		color_image: Arc<ImageView>,
	) -> crate::Result<()>
	{
		let sky_render_info = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::DontCare,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(color_image)
			})],
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		};
		let sky_cb = self.command_buffer.take().unwrap();
		cb_builder
			.begin_rendering(sky_render_info)?
			.execute_commands(sky_cb)?
			.end_rendering()?;

		Ok(())
	}
}
