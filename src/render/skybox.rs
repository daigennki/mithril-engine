 /* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::path::Path;
use vulkano::buffer::{ ImmutableBuffer, BufferUsage };
use vulkano::pipeline::graphics::{ input_assembly::PrimitiveTopology, depth_stencil::CompareOp };
use vulkano::descriptor_set::{ PersistentDescriptorSet, WriteDescriptorSet };
use vulkano::sampler::SamplerCreateInfo;

use super::{ RenderContext, command_buffer::CommandBuffer, texture::CubemapTexture };
use crate::GenericEngineError;

#[derive(shipyard::Unique)]
pub struct Skybox
{
	sky_pipeline: super::pipeline::Pipeline,
	cube_vbo: Arc<ImmutableBuffer<[f32]>>,
	cube_ibo: Arc<ImmutableBuffer<[u16]>>,
	descriptor_set: Arc<PersistentDescriptorSet>
}
impl Skybox
{
	pub fn new(render_ctx: &mut RenderContext) -> Result<Self, GenericEngineError>
	{
		// sky pipeline
		// TODO: this should be moved out of here so we're not creating it again when the skybox is changed
		let render_pass = render_ctx.get_current_framebuffer().render_pass().clone();
		let dim = render_ctx.swapchain_dimensions();
		let sampler_info = SamplerCreateInfo::simple_repeat_linear_no_mipmap();
		let cubemap_sampler = vulkano::sampler::Sampler::new(render_ctx.get_queue().device().clone(), sampler_info)?;
		let sky_pipeline = super::pipeline::Pipeline::new(
			PrimitiveTopology::TriangleStrip,
			"skybox.vert.spv".into(), Some("skybox.frag.spv".into()),
			vec![ (0, 0, cubemap_sampler) ],
			render_pass,
			dim[0], dim[1],
			CompareOp::Always
		)?;

		// sky texture cubemap
		let sky_cubemap = render_ctx.new_cubemap_texture([
			Path::new("sky/Daylight Box_Right.png"),
			Path::new("sky/Daylight Box_Left.png"),
			Path::new("sky/Daylight Box_Top.png"),
			Path::new("sky/Daylight Box_Bottom.png"),
			Path::new("sky/Daylight Box_Front.png"),
			Path::new("sky/Daylight Box_Back.png")
		])?;
		let descriptor_set = sky_pipeline.new_descriptor_set(0, [
			WriteDescriptorSet::image_view(1, sky_cubemap.view().clone())
		])?;

		// sky cube
		let indices: [u16; 20] = [
			1, 0, 3, 2,				// +Y quad	(+X between +Y and -Y)
			5, 4, 7, 6,				// -Y quad
			1, 0, u16::MAX,		// finish -X
			7, 1, 5, 3, u16::MAX,	// -Z quad
			0, 6, 2, 4				// +Z quad
		];
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
			sky_pipeline: sky_pipeline,
			cube_vbo: render_ctx.new_buffer_from_iter(position, BufferUsage::vertex_buffer())?,
			cube_ibo: render_ctx.new_buffer_from_iter(indices, BufferUsage::index_buffer())?,
			descriptor_set: descriptor_set
		})
	}

	// TODO: add a function for resizing viewport

	pub fn draw<L>(&self, cb: &mut CommandBuffer<L>, camera: &crate::component::camera::Camera) -> Result<(), GenericEngineError>
	{
		cb.bind_pipeline(&self.sky_pipeline);
		cb.bind_descriptor_set(0, vec![ self.descriptor_set.clone() ])?;
		camera.bind(cb)?;
		cb.bind_vertex_buffers(0, vec![ self.cube_vbo.clone() ]);
		cb.bind_index_buffer(self.cube_ibo.clone());
		cb.draw_indexed(20, 1, 0, 0, 0)?;
		Ok(())
	}
}
