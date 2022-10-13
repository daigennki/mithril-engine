/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, DeviceLocalBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::graphics::{depth_stencil::CompareOp, input_assembly::PrimitiveTopology};
use vulkano::sampler::SamplerCreateInfo;

use super::{command_buffer::CommandBuffer, RenderContext};
use crate::GenericEngineError;

#[derive(shipyard::Unique)]
pub struct Skybox
{
	sky_pipeline: super::pipeline::Pipeline,
	cube_vbo: Arc<DeviceLocalBuffer<[f32]>>,
	cube_ibo: Arc<DeviceLocalBuffer<[u16]>>,
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
		let render_pass = render_ctx.get_current_framebuffer().render_pass().clone();
		let sampler_info = SamplerCreateInfo::simple_repeat_linear_no_mipmap();
		let cubemap_sampler = vulkano::sampler::Sampler::new(render_ctx.get_queue().device().clone(), sampler_info)?;
		let sky_pipeline = super::pipeline::Pipeline::new(
			PrimitiveTopology::TriangleStrip,
			"skybox.vert.spv".into(),
			Some("skybox.frag.spv".into()),
			vec![(0, 0, cubemap_sampler)],
			render_pass,
			CompareOp::Always,
		)?;

		// sky texture cubemap
		let face_names = [
			"Right", "Left", "Top", "Bottom", "Front", "Back",
		];
		let face_paths = face_names.map(|face_name| tex_files_format.replace('*', face_name).into());
		let sky_cubemap = render_ctx.new_cubemap_texture(face_paths)?;
		let descriptor_set =
			sky_pipeline.new_descriptor_set(0, [WriteDescriptorSet::image_view(1, sky_cubemap.view().clone())])?;

		// sky cube
		#[rustfmt::skip]
		let indices: [u16; 20] = [
			1, 0, 3, 2,				// +Y quad	(+X between +Y and -Y)
			5, 4, 7, 6,				// -Y quad
			1, 0, u16::MAX,		// finish -X
			7, 1, 5, 3, u16::MAX,	// -Z quad
			0, 6, 2, 4				// +Z quad
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
			cube_vbo: render_ctx.new_buffer_from_iter(position, BufferUsage { vertex_buffer: true, ..BufferUsage::empty() })?,
			cube_ibo: render_ctx.new_buffer_from_iter(indices, BufferUsage { index_buffer: true, ..BufferUsage::empty() })?,
			descriptor_set,
		})
	}

	pub fn draw<L>(
		&self, cb: &mut CommandBuffer<L>, camera: &crate::component::camera::Camera,
	) -> Result<(), GenericEngineError>
	{
		cb.bind_pipeline(&self.sky_pipeline);
		cb.bind_descriptor_set(0, vec![self.descriptor_set.clone()])?;
		camera.bind(cb)?;
		cb.bind_vertex_buffers(0, vec![self.cube_vbo.clone()]);
		cb.bind_index_buffer(self.cube_ibo.clone());
		cb.draw_indexed(20, 1, 0, 0, 0)?;
		Ok(())
	}
}
