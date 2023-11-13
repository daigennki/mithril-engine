/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;
use glam::*;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::pipeline::{layout::PushConstantRange, Pipeline, PipelineBindPoint};
use vulkano::pipeline::graphics::{
	color_blend::{ColorBlendState, ColorBlendAttachmentState},
	depth_stencil::DepthStencilState,
	input_assembly::PrimitiveTopology,
	rasterization::RasterizationState,
	subpass::PipelineRenderingCreateInfo,
	GraphicsPipeline,
};
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::shader::ShaderStages;

use super::RenderContext;
use crate::GenericEngineError;

mod vs {
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
mod fs {
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
	pub fn new(render_ctx: &mut RenderContext, tex_files_format: String) -> Result<Self, GenericEngineError>
	{
		let rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![ Some(Format::R16G16B16A16_SFLOAT), ],
			depth_attachment_format: Some(super::MAIN_DEPTH_FORMAT),
			..Default::default()
		};

		let device = render_ctx.descriptor_set_allocator().device().clone();

		let cubemap_sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear_no_mipmap())?;
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, DescriptorSetLayoutBinding { // binding 0: skybox cubemap texture and sampler 
					stages: ShaderStages::FRAGMENT,
					immutable_samplers: vec![ cubemap_sampler ],
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
				}),
			].into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

		let push_constant_range = PushConstantRange { // push constant for projview matrix
			stages: ShaderStages::VERTEX,
			offset: 0,
			size: std::mem::size_of::<Mat4>().try_into().unwrap(),
		};

		let sky_pipeline = super::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleFan,
			&[ vs::load(device.clone())?, fs::load(device.clone())? ],
			RasterizationState::default(),
			Some(ColorBlendState::with_attachment_states(1, ColorBlendAttachmentState::default())),
			vec![ set_layout.clone() ],
			vec![ push_constant_range ],
			rendering_info,
			Some(DepthStencilState::default()),
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

		// sky cube, consisting of two fans with the "center" being opposite corners of the cube
		#[rustfmt::skip]
		let indices: [u16; 17] = [
			0, 1, 2, 3, 4, 5, 6, 1, u16::MAX,
			7, 1, 2, 3, 4, 5, 6, 1,
		];

		// relative to camera at default state, -X is left, +Y is forward, and +Z is up
		#[rustfmt::skip]
		let position: [f32; 24] = [
			-1.0, -1.0, -1.0,
			-1.0, -1.0, 1.0,
			1.0, -1.0, 1.0,
			1.0, -1.0, -1.0,
			1.0, 1.0, -1.0,
			-1.0, 1.0, -1.0,
			-1.0, 1.0, 1.0,
			1.0, 1.0, 1.0,
		];

		Ok(Skybox {
			sky_pipeline,
			cube_vbo: render_ctx.new_buffer(position, BufferUsage::VERTEX_BUFFER)?,
			cube_ibo: render_ctx.new_buffer(indices, BufferUsage::INDEX_BUFFER)?,
			descriptor_set,
		})
	}

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		sky_projview: Mat4,
	) -> Result<(), GenericEngineError>
	{
		cb
			.bind_pipeline_graphics(self.sky_pipeline.clone())?
			.bind_descriptor_sets(
				PipelineBindPoint::Graphics,
				self.sky_pipeline.layout().clone(),
				0,
				vec![self.descriptor_set.clone()]
			)?
			.push_constants(self.sky_pipeline.layout().clone(), 0, sky_projview)?
			.bind_vertex_buffers(0, vec![self.cube_vbo.clone()])?
			.bind_index_buffer(self.cube_ibo.clone())?
			.draw_indexed(17, 1, 0, 0, 0)?;
		Ok(())
	}
}
