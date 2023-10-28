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
use vulkano::pipeline::{layout::PushConstantRange, PipelineBindPoint};
use vulkano::pipeline::graphics::{ 
	color_blend::{ColorBlendState, ColorBlendAttachmentState}, depth_stencil::CompareOp, input_assembly::PrimitiveTopology,
	subpass::PipelineRenderingCreateInfo
};
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::shader::ShaderStages;

use super::RenderContext;
use crate::GenericEngineError;

mod vs {
	vulkano_shaders::shader! {
		ty: "vertex",
		bytes: "shaders/skybox.vert.spv",
	}
}
mod fs {
	vulkano_shaders::shader! {
		ty: "fragment",
		bytes: "shaders/skybox.frag.spv",
	}
}

#[derive(shipyard::Unique)]
pub struct Skybox
{
	sky_pipeline: super::pipeline::Pipeline,
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
		// sky pipeline
		// TODO: this should be moved out of here so we're not creating it again when the skybox is changed
		let sampler_info = SamplerCreateInfo::simple_repeat_linear_no_mipmap();
		let cubemap_sampler = Sampler::new(render_ctx.get_queue().device().clone(), sampler_info)?;

		let rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![ Some(Format::R16G16B16A16_SFLOAT), ],
			depth_attachment_format: Some(Format::D16_UNORM),
			..Default::default()
		};

		let device = render_ctx.descriptor_set_allocator().device().clone();
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, DescriptorSetLayoutBinding { // binding 0: sampler0
					stages: ShaderStages::FRAGMENT,
					immutable_samplers: vec![ cubemap_sampler ],
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
				}),
				(1, DescriptorSetLayoutBinding { // binding 1: skybox texture
					stages: ShaderStages::FRAGMENT,
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
				}),
			].into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

		let sky_pipeline = super::pipeline::Pipeline::new_from_binary(
			device.clone(),
			PrimitiveTopology::TriangleStrip,
			vs::load(device.clone())?,
			Some((
				fs::load(device.clone())?,
				ColorBlendState::with_attachment_states(1, ColorBlendAttachmentState::default())
			)),
			vec![ set_layout.clone() ],
			vec![ 
				PushConstantRange { // push constant for view matrix
					stages: ShaderStages::VERTEX,
					offset: 0,
					size: std::mem::size_of::<Mat4>().try_into().unwrap(),
				}
			],
			rendering_info,
			CompareOp::Always,
			true,
		)?;

		// sky texture cubemap
		let face_names = ["Right", "Left", "Top", "Bottom", "Front", "Back"];
		let face_paths = face_names.map(|face_name| tex_files_format.replace('*', face_name).into());
		let sky_cubemap = render_ctx.new_cubemap_texture(face_paths)?;

		let descriptor_set = PersistentDescriptorSet::new(
			render_ctx.descriptor_set_allocator(),
			set_layout,
			[WriteDescriptorSet::image_view(1, sky_cubemap.view().clone())],
			[],
		)?;

		// sky cube
		#[rustfmt::skip]
		let indices: [u16; 20] = [
			0, 1, 2, 3,				// +Y quad	(+X between +Y and -Y)
			4, 5, 6, 7,				// -Y quad
			0, 1, u16::MAX,		// finish -X
			1, 7, 3, 5, u16::MAX,	// -Z quad
			0, 2, 6, 4				// +Z quad
		];
		#[rustfmt::skip]
		let position: [f32; 24] = [
			-1.0, 1.0, 1.0,		// relative to camera at default state, -X is left, +Y is forward, and +Z is up
			-1.0, 1.0, -1.0,
			1.0, 1.0, 1.0,
			1.0, 1.0, -1.0,
			1.0, -1.0, 1.0,
			1.0, -1.0, -1.0,
			-1.0, -1.0, 1.0,
			-1.0, -1.0, -1.0
		];

		Ok(Skybox {
			sky_pipeline,
			cube_vbo: render_ctx.new_buffer_from_iter(position, BufferUsage::VERTEX_BUFFER)?,
			cube_ibo: render_ctx.new_buffer_from_iter(indices, BufferUsage::INDEX_BUFFER)?,
			descriptor_set,
		})
	}

	pub fn draw(
		&self,
		cb: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
		sky_projview: Mat4,
	) -> Result<(), GenericEngineError>
	{
		self.sky_pipeline.bind(cb)?;
		cb.bind_descriptor_sets(PipelineBindPoint::Graphics, self.sky_pipeline.layout(), 0, vec![self.descriptor_set.clone()])?;
		cb.push_constants(self.sky_pipeline.layout(), 0, sky_projview)?;
		cb.bind_vertex_buffers(0, vec![self.cube_vbo.clone()])?;
		cb.bind_index_buffer(self.cube_ibo.clone())?;
		cb.draw_indexed(20, 1, 0, 0, 0)?;
		Ok(())
	}
}
