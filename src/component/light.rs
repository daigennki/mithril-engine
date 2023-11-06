/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use shipyard::{Get, IntoIter, IntoWorkloadSystem, UniqueView, UniqueViewMut, View, WorkloadSystem};
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{
	sampler::{Filter, SamplerCreateInfo, Sampler}, view::ImageView, Image, ImageCreateInfo, ImageUsage
};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::pipeline::{
	graphics::{
		depth_stencil::{CompareOp, DepthState, DepthStencilState},
		input_assembly::PrimitiveTopology, rasterization::{RasterizationState, CullMode, DepthBiasState},
		subpass::PipelineRenderingCreateInfo, GraphicsPipeline,
	},
	layout::PushConstantRange,
};
use vulkano::shader::ShaderStages;

use crate::{render::RenderContext, GenericEngineError};
use super::{EntityComponent, WantsSystemAdded, Transform, camera::CameraManager};

/// These are various components that represent light sources in the world.
/// 

#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct DirectionalLight
{
	pub color: Vec3,
	pub intensity: f32,
}
impl WantsSystemAdded for DirectionalLight
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		Some(update_directional_light.into_workload_system().unwrap())
	}
}
fn update_directional_light(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut light_manager: UniqueViewMut<LightManager>,
	camera_manager: UniqueView<CameraManager>,
	dir_lights: View<DirectionalLight>,
	transforms: View<Transform>,
)
{
	// Only use the first directional light found.
	// There should really be only one of these in the world anyways.
	if let Some((dl, t)) = (&dir_lights, &transforms).iter().next() {
		match transforms.get(camera_manager.active_camera()) {
			Ok(camera_transform) => {
				if let Err(e) = light_manager.update_dir_light(&mut render_ctx, dl, t, camera_transform.position) {
					log::error!("update_directional_light: Failed to update `DirectionalLight` GPU buffer: {}", e);
				}
			}
			Err(e) => log::error!("update_directional_light: {} (active camera ID might be invalid or not set)", e),
		}
	}
}

/*#[derive(shipyard::Component, Deserialze, EntityComponent)]
pub struct PointLight
{
	pub color: Vec3,
	pub intensity: f32,
}

#[derive(shipyard::Component, Deserialze, EntityComponent)]
pub struct SpotLight
{
	pub color: Vec3,
	pub intensity: f32,
}*/


#[derive(Clone, Copy, bytemuck::AnyBitPattern)]
#[repr(C)]
struct DirLightData
{
	projview: Mat4,
	direction: Vec3A,
	color_intensity: Vec4, // RGB is color, A is intensity
}
impl Default for DirLightData
{
	fn default() -> Self
	{
		DirLightData {
			projview: Mat4::IDENTITY,
			direction: Vec3A::NEG_Z,
			color_intensity: Vec4::ONE,
		}
	}
}

/*struct PointLightData
{
	pub position: Vec3A,
	pub color_intensity: Vec4,
}
struct SpotLightData
{
	pub position: Vec3A,
	pub direction: Vec3A,
	pub color_intensity: Vec4,
}*/

#[derive(shipyard::Unique)]
pub struct LightManager
{
	dir_light_projview: Mat4,
	dir_light_buf: Subbuffer<[DirLightData]>,
	dir_light_shadow: Arc<ImageView>,
	dir_light_cb: Option<Arc<SecondaryAutoCommandBuffer>>,

	/*point_light_buf: Arc<Subbuffer<[PointLightData]>>,
	spot_light_buf: Arc<Subbuffer<[SpotLightData]>>,*/

	all_lights_set: Arc<PersistentDescriptorSet>,

	shadow_pipeline: Arc<GraphicsPipeline>,
}
impl LightManager
{
	pub fn new(render_ctx: &mut RenderContext) -> Result<Self, GenericEngineError>
	{
		let device = render_ctx.descriptor_set_allocator().device().clone();

		/* info for shadow images */
		let image_info = ImageCreateInfo {
			usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
			format: Format::D16_UNORM,
			extent: [ 1024, 1024, 1 ],
			..Default::default()
		};

		/* directional light */
		let dir_light_data = DirLightData::default();
		let dir_light_buf = render_ctx.new_buffer([dir_light_data], BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST)?;

		let dir_light_shadow_img = Image::new(
			render_ctx.memory_allocator().clone(), 
			image_info.clone(), 
			AllocationCreateInfo::default()
		)?;
		let dir_light_shadow_view = ImageView::new_default(dir_light_shadow_img)?;

		/* shadow sampler */
		let sampler_info = SamplerCreateInfo {
			mag_filter: Filter::Linear,
			min_filter: Filter::Linear,
			compare: Some(CompareOp::LessOrEqual),
			..Default::default()
		};
		let shadow_sampler = Sampler::new(device.clone(), sampler_info)?;

		/* descriptor set with everything lighting- and shadow-related */
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, DescriptorSetLayoutBinding { // shadow sampler
					stages: ShaderStages::FRAGMENT,
					immutable_samplers: vec![ shadow_sampler ],
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
				}),
				(1, DescriptorSetLayoutBinding { // directional light buffer 
					stages: ShaderStages::FRAGMENT,
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
				}),
				(2, DescriptorSetLayoutBinding { // directional light shadow
					stages: ShaderStages::FRAGMENT,
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
				}),
			].into(),
			..Default::default()
		};
		let all_lights_set = PersistentDescriptorSet::new(
			render_ctx.descriptor_set_allocator(),
			DescriptorSetLayout::new(device.clone(), set_layout_info)?,
			[
				WriteDescriptorSet::buffer(1, dir_light_buf.clone()),
				WriteDescriptorSet::image_view(2, dir_light_shadow_view.clone()),
			],
			[]
		)?;

		/* shadow pipeline */
		let depth_stencil_state = DepthStencilState {
			depth: Some(DepthState::simple()),
			..Default::default()
		};
		let rasterization_state = RasterizationState {
			cull_mode: CullMode::Back,
			depth_bias: Some(DepthBiasState::default()),
			..Default::default()
		};
		let shadow_rendering = PipelineRenderingCreateInfo {
			depth_attachment_format: Some(Format::D16_UNORM),
			..Default::default()
		};
		let shadow_pipeline = crate::render::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleList,
			&[ vs_shadow::load(device.clone())? ],
			rasterization_state,
			None,
			vec![],
			vec![
				PushConstantRange { // push constant for transformation matrix
					stages: ShaderStages::VERTEX,
					offset: 0,
					size: std::mem::size_of::<Mat4>().try_into().unwrap(),
				}
			],
			shadow_rendering,
			Some(depth_stencil_state),
		)?;

		Ok(LightManager {
			dir_light_projview: Mat4::IDENTITY,
			dir_light_buf,
			dir_light_shadow: dir_light_shadow_view,
			dir_light_cb: None,
			all_lights_set,
			shadow_pipeline,
		})
	}

	pub fn update_dir_light(
		&mut self,
		render_ctx: &mut RenderContext,
		light: &DirectionalLight,
		transform: &Transform,
		camera_pos: Vec3,
	)
		-> Result<(), GenericEngineError>
	{
		let direction = transform.rotation_quat() * Vec3A::NEG_Z;
		let dir_vec3 = direction.into();
		let far = 100.0;
		let proj = Mat4::orthographic_lh(-10.0, 10.0, -10.0, 10.0, 1.0, far);

		// TODO: adjust the eye position here to more efficiently cover the camera frustum
		let view = Mat4::look_to_lh(camera_pos - dir_vec3 * 50.0, dir_vec3, Vec3::NEG_Y);

		self.dir_light_projview = proj * view;

		let dir_light_data = DirLightData {
			projview: self.dir_light_projview,
			direction,
			color_intensity: light.color.extend(light.intensity),
		};
		render_ctx.copy_to_buffer([dir_light_data], self.dir_light_buf.clone())?;
		Ok(())
	}

	pub fn add_dir_light_cb(&mut self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		self.dir_light_cb = Some(cb);
	}
	pub fn take_dir_light_cb(&mut self) -> Option<Arc<SecondaryAutoCommandBuffer>>
	{
		self.dir_light_cb.take()
	}

	pub fn get_dir_light_shadow(&self) -> &Arc<ImageView>
	{
		&self.dir_light_shadow
	}
	pub fn get_dir_light_projview(&self) -> Mat4
	{
		self.dir_light_projview
	}

	pub fn get_all_lights_set(&self) -> &Arc<PersistentDescriptorSet>
	{
		&self.all_lights_set
	}

	pub fn get_shadow_pipeline(&self) -> &Arc<GraphicsPipeline>
	{
		&self.shadow_pipeline
	}
}

mod vs_shadow
{
	vulkano_shaders::shader! {
		ty: "vertex",
		src: r"
			#version 450

			layout(push_constant) uniform pc
			{
				mat4 projviewmodel;	// pre-multiplied projection, view, and model transformation matrices
			};

			layout(location = 0) in vec3 position;

			void main()
			{
				gl_Position = projviewmodel * vec4(position, 1.0);
			}
		",
	}
}

