/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::pipeline::graphics::{input_assembly::PrimitiveTopology, rasterization::RasterizationState, GraphicsPipeline};
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{Pipeline, PipelineBindPoint, PipelineLayout};
use vulkano::shader::ShaderStages;

use super::RenderContext;
use crate::EngineError;

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
	pub fn new(render_ctx: &mut RenderContext, tex_files_format: String) -> Result<Self, EngineError>
	{
		let device = render_ctx.descriptor_set_allocator().device().clone();

		let cubemap_sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear_no_mipmap())
			.map_err(|e| EngineError::vulkan_error("failed to create sampler", e))?;
		let tex_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			immutable_samplers: vec![cubemap_sampler],
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
		};
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [(0, tex_binding)].into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)
			.map_err(|e| EngineError::vulkan_error("failed to create descriptor set layout", e))?;

		let pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![set_layout.clone()],
			push_constant_ranges: vec![PushConstantRange {
				stages: ShaderStages::VERTEX,
				offset: 0,
				size: std::mem::size_of::<Mat4>().try_into().unwrap(),
			}],
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_info)
			.map_err(|e| EngineError::vulkan_error("failed to create pipeline layout", e))?;

		let sky_pipeline = super::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleFan,
			&[vs::load(device.clone()).unwrap(), fs::load(device.clone()).unwrap()],
			RasterizationState::default(),
			pipeline_layout,
			&[(Format::R16G16B16A16_SFLOAT, None)],
			None,
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
		)
		.map_err(|e| EngineError::vulkan_error("failed to create descriptor set", e))?;

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
			cube_vbo: render_ctx.new_buffer(&position, BufferUsage::VERTEX_BUFFER)?,
			cube_ibo: render_ctx.new_buffer(&indices, BufferUsage::INDEX_BUFFER)?,
			descriptor_set,
			command_buffer: None,
		})
	}

	pub fn draw(&mut self, render_ctx: &RenderContext, sky_projview: Mat4) -> Result<(), EngineError>
	{
		let vp_extent = render_ctx.swapchain_dimensions();
		let mut cb = render_ctx.gather_commands(&[Format::R16G16B16A16_SFLOAT], None, None, vp_extent)?;

		cb.bind_pipeline_graphics(self.sky_pipeline.clone())
			.unwrap()
			.bind_descriptor_sets(
				PipelineBindPoint::Graphics,
				self.sky_pipeline.layout().clone(),
				0,
				vec![self.descriptor_set.clone()],
			)
			.unwrap()
			.push_constants(self.sky_pipeline.layout().clone(), 0, sky_projview)
			.unwrap()
			.bind_vertex_buffers(0, vec![self.cube_vbo.clone()])
			.unwrap()
			.bind_index_buffer(self.cube_ibo.clone())
			.unwrap()
			.draw_indexed(17, 1, 0, 0, 0)
			.unwrap();

		let built_cb = cb
			.build()
			.map_err(|e| EngineError::vulkan_error("failed to build command buffer", e))?;

		assert!(self.command_buffer.replace(built_cb).is_none());

		Ok(())
	}

	pub fn take_cb(&mut self) -> Option<Arc<SecondaryAutoCommandBuffer>>
	{
		self.command_buffer.take()
	}
}
