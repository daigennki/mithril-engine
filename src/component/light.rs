/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use shipyard::{IntoIter, IntoWorkloadSystem, UniqueView, UniqueViewMut, View, WorkloadSystem};
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{
	sampler::{Filter, Sampler, SamplerCreateInfo},
	view::{ImageView, ImageViewCreateInfo},
	Image, ImageAspects, ImageCreateInfo, ImageSubresourceRange, ImageUsage,
};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::pipeline::{
	graphics::{
		depth_stencil::{CompareOp, DepthState, DepthStencilState},
		input_assembly::PrimitiveTopology,
		rasterization::{CullMode, DepthBiasState, RasterizationState},
		subpass::PipelineRenderingCreateInfo,
		GraphicsPipeline,
	},
	layout::PushConstantRange,
};
use vulkano::shader::ShaderStages;

use super::{camera::CameraManager, EntityComponent, Transform, WantsSystemAdded};
use crate::{render::RenderContext, GenericEngineError};

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
		None
	}
	fn add_prerender_system(&self) -> Option<WorkloadSystem>
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
		// Cut the camera frustum into different pieces for the light.
		let fars = [6.0, 12.0, 24.0];
		let mut cut_frustums: [Mat4; 3] = Default::default();
		let mut near = crate::component::camera::CAMERA_NEAR;
		for (i, far) in fars.into_iter().enumerate() {
			cut_frustums[i] = camera_manager.proj_with_near_far(near, far);
			near = fars[i];
		}

		if let Err(e) = light_manager.update_dir_light(&mut render_ctx, dl, t, cut_frustums) {
			log::error!(
				"update_directional_light: Failed to update `DirectionalLight` GPU buffer: {}",
				e
			);
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
	projviews: [Mat4; 3],
	direction: Vec3A,
	color_intensity: Vec4, // RGB is color, A is intensity
}
impl Default for DirLightData
{
	fn default() -> Self
	{
		DirLightData {
			projviews: Default::default(),
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
	dir_light_projviews: [Mat4; 3],
	dir_light_buf: Subbuffer<[DirLightData]>,
	dir_light_shadow: Arc<ImageView>,
	dir_light_shadow_layers: Vec<Arc<ImageView>>,
	dir_light_cb: Vec<Arc<SecondaryAutoCommandBuffer>>,

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
			extent: [1024, 1024, 1],
			array_layers: 3,
			..Default::default()
		};

		/* directional light */
		let dir_light_data = DirLightData::default();
		let dir_light_buf =
			render_ctx.new_buffer(&[dir_light_data], BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST)?;

		let dir_light_shadow_img = Image::new(
			render_ctx.memory_allocator().clone(),
			image_info.clone(),
			AllocationCreateInfo::default(),
		)?;
		let dir_light_shadow_view_info = ImageViewCreateInfo {
			usage: ImageUsage::SAMPLED,
			..ImageViewCreateInfo::from_image(&dir_light_shadow_img)
		};
		let dir_light_shadow_view = ImageView::new(dir_light_shadow_img.clone(), dir_light_shadow_view_info)?;

		let mut dir_light_shadow_layers = Vec::with_capacity(dir_light_shadow_img.array_layers().try_into().unwrap());
		for i in 0..dir_light_shadow_img.array_layers() {
			let layer_info = ImageViewCreateInfo {
				subresource_range: ImageSubresourceRange {
					aspects: ImageAspects::DEPTH,
					mip_levels: 0..1,
					array_layers: i..(i + 1),
				},
				usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
				..ImageViewCreateInfo::from_image(&dir_light_shadow_img)
			};
			let layer_view = ImageView::new(dir_light_shadow_img.clone(), layer_info)?;
			dir_light_shadow_layers.push(layer_view);
		}

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
				(
					0,
					DescriptorSetLayoutBinding {
						// shadow sampler
						stages: ShaderStages::FRAGMENT,
						immutable_samplers: vec![shadow_sampler],
						..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
					},
				),
				(
					1,
					DescriptorSetLayoutBinding {
						// directional light buffer
						stages: ShaderStages::FRAGMENT,
						..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
					},
				),
				(
					2,
					DescriptorSetLayoutBinding {
						// directional light shadow
						stages: ShaderStages::FRAGMENT,
						..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
					},
				),
			]
			.into(),
			..Default::default()
		};
		let all_lights_set = PersistentDescriptorSet::new(
			render_ctx.descriptor_set_allocator(),
			DescriptorSetLayout::new(device.clone(), set_layout_info)?,
			[
				WriteDescriptorSet::buffer(1, dir_light_buf.clone()),
				WriteDescriptorSet::image_view(2, dir_light_shadow_view.clone()),
			],
			[],
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
			&[vs_shadow::load(device.clone())?],
			rasterization_state,
			None,
			vec![],
			vec![PushConstantRange {
				// push constant for transformation matrix
				stages: ShaderStages::VERTEX,
				offset: 0,
				size: std::mem::size_of::<Mat4>().try_into().unwrap(),
			}],
			shadow_rendering,
			Some(depth_stencil_state),
		)?;

		Ok(LightManager {
			dir_light_projviews: Default::default(),
			dir_light_buf,
			dir_light_shadow: dir_light_shadow_view,
			dir_light_shadow_layers,
			dir_light_cb: Vec::with_capacity(dir_light_shadow_img.array_layers().try_into().unwrap()),
			all_lights_set,
			shadow_pipeline,
		})
	}

	pub fn update_dir_light(
		&mut self,
		render_ctx: &mut RenderContext,
		light: &DirectionalLight,
		transform: &Transform,
		cut_camera_frustums: [Mat4; 3],
	) -> Result<(), GenericEngineError>
	{
		let direction = transform.rotation_quat() * Vec3A::NEG_Z;

		// Fit the light view and projection matrices to different sections of the camera frustum.
		// Most of this is adapted from here: https://learnopengl.com/Guest-Articles/2021/CSM
		for (i, cut_frustum) in cut_camera_frustums.iter().enumerate() {
			let camera_projview_inv = cut_frustum.inverse();
			let mut frustum_corners: [Vec4; 8] = Default::default();
			let mut corner_i = 0;
			for x in 0..2 {
				for y in 0..2 {
					for z in 0..2 {
						let pt_x = 2 * x - 1;
						let pt_y = 2 * y - 1;
						let pt = camera_projview_inv * IVec4::new(pt_x, pt_y, z, 1).as_vec4();
						frustum_corners[corner_i] = pt / pt.w;
						corner_i += 1;
					}
				}
			}

			let center = frustum_corners.iter().sum::<Vec4>() * (1.0 / 8.0);
			let view = Mat4::look_to_lh(center.truncate(), direction.into(), Vec3::Y);

			let mut min_x = f32::MAX;
			let mut max_x = f32::MIN;
			let mut min_y = f32::MAX;
			let mut max_y = f32::MIN;
			let mut min_z = f32::MAX;
			let mut max_z = f32::MIN;
			for v in frustum_corners {
				let trf = view * v;
				min_x = min_x.min(trf.x);
				max_x = max_x.max(trf.x);
				min_y = min_y.min(trf.y);
				max_y = max_y.max(trf.y);
				min_z = min_z.min(trf.z);
				max_z = max_z.max(trf.z);
			}

			let z_mul = 10.0;
			if min_z < 0.0 {
				min_z *= z_mul;
			} else {
				min_z /= z_mul;
			}
			if max_z < 0.0 {
				max_z /= z_mul;
			} else {
				max_z *= z_mul;
			}

			let proj = Mat4::orthographic_lh(min_x, max_x, min_y, max_y, min_z, max_z);

			self.dir_light_projviews[i] = proj * view;
		}

		let dir_light_data = DirLightData {
			projviews: self.dir_light_projviews,
			direction,
			color_intensity: light.color.extend(light.intensity),
		};
		render_ctx.update_buffer(&[dir_light_data], self.dir_light_buf.clone())?;
		Ok(())
	}

	pub fn add_dir_light_cb(&mut self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		if self.dir_light_cb.len() == self.dir_light_cb.capacity() {
			panic!("attempted to add too many command buffers for directional light rendering");
		}
		self.dir_light_cb.push(cb);
	}
	pub fn drain_dir_light_cb(&mut self) -> Vec<(Arc<SecondaryAutoCommandBuffer>, Arc<ImageView>)>
	{
		self.dir_light_cb
			.drain(..)
			.zip(self.dir_light_shadow_layers.iter().cloned())
			.collect()
	}

	pub fn get_dir_light_shadow(&self) -> &Arc<ImageView>
	{
		&self.dir_light_shadow
	}
	pub fn get_dir_light_projviews(&self) -> [Mat4; 3]
	{
		self.dir_light_projviews
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
