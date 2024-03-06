/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use glam::*;
use std::sync::{Arc, Mutex};
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::*;
use vulkano::descriptor_set::{allocator::*, layout::*, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{sampler::*, view::*, *};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::pipeline::graphics::{
	depth_stencil::*, rasterization::*, subpass::PipelineRenderingCreateInfo, vertex_input::VertexInputState, *,
};
use vulkano::pipeline::{layout::*, *};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

use crate::component::light::DirectionalLight;
use crate::component::Transform;

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct DirLightData
{
	projviews: [Mat4; 3],
	direction: Vec4,       // "w" element only exists for alignment and will be set to 0
	color_intensity: Vec4, // RGB is color, A is intensity
}
impl Default for DirLightData
{
	fn default() -> Self
	{
		DirLightData {
			projviews: Default::default(),
			direction: Vec4::NEG_Z,
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
	dir_light_projviews: [DMat4; 3],
	dir_light_buf: Subbuffer<[DirLightData]>,
	dir_light_shadow: Arc<ImageView>,
	dir_light_shadow_layers: Vec<Arc<ImageView>>,
	dir_light_cb: Mutex<Vec<Arc<SecondaryAutoCommandBuffer>>>,

	/*point_light_buf: Arc<Subbuffer<[PointLightData]>>,
	spot_light_buf: Arc<Subbuffer<[SpotLightData]>>,*/
	all_lights_set: Arc<PersistentDescriptorSet>,

	shadow_pipeline: Arc<GraphicsPipeline>,

	update_needed: Option<DirLightData>,
}
impl LightManager
{
	pub fn new(render_ctx: &mut super::RenderContext) -> crate::Result<Self>
	{
		let device = render_ctx.memory_allocator.device().clone();
		let set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 2,
			..Default::default()
		};
		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), set_alloc_info);

		/* info for shadow images */
		let image_info = ImageCreateInfo {
			usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
			format: Format::D16_UNORM,
			extent: [1024, 1024, 1],
			array_layers: 3,
			..Default::default()
		};

		/* descriptor set with everything lighting- and shadow-related */
		let shadow_sampler_info = SamplerCreateInfo {
			mag_filter: Filter::Linear,
			min_filter: Filter::Linear,
			compare: Some(CompareOp::LessOrEqual),
			..Default::default()
		};
		let shadow_sampler = Sampler::new(device.clone(), shadow_sampler_info)?;
		let light_bindings = [
			DescriptorSetLayoutBinding {
				// binding 0: shadow sampler
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![shadow_sampler],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
			},
			DescriptorSetLayoutBinding {
				// binding 1: directional light buffer
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
			},
			DescriptorSetLayoutBinding {
				// binding 2: directional light shadow
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
			},
		];
		let light_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: (0..).zip(light_bindings).collect(),
			..Default::default()
		};
		let light_set_layout = DescriptorSetLayout::new(device.clone(), light_set_layout_info)?;

		/* directional light */
		let dir_light_data = DirLightData::default();
		let dir_light_buf =
			render_ctx.new_buffer(&[dir_light_data], BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST)?;

		let dir_light_shadow_img = Image::new(
			render_ctx.memory_allocator.clone(),
			image_info.clone(),
			AllocationCreateInfo::default(),
		)?;
		let dir_light_shadow_view_info = ImageViewCreateInfo {
			usage: ImageUsage::SAMPLED,
			..ImageViewCreateInfo::from_image(&dir_light_shadow_img)
		};
		let dir_light_shadow_view = ImageView::new(dir_light_shadow_img.clone(), dir_light_shadow_view_info)?;

		let dir_light_shadow_layers = (0..dir_light_shadow_img.array_layers())
			.map(|i| {
				let layer_info = ImageViewCreateInfo {
					subresource_range: ImageSubresourceRange {
						aspects: ImageAspects::DEPTH,
						mip_levels: 0..1,
						array_layers: i..(i + 1),
					},
					usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
					..ImageViewCreateInfo::from_image(&dir_light_shadow_img)
				};
				ImageView::new(dir_light_shadow_img.clone(), layer_info)
			})
			.collect::<Result<_, _>>()?;

		/* descriptor set with everything lighting- and shadow-related */
		let all_lights_set = PersistentDescriptorSet::new(
			&descriptor_set_allocator,
			light_set_layout.clone(),
			[
				WriteDescriptorSet::buffer(1, dir_light_buf.clone()),
				WriteDescriptorSet::image_view(2, dir_light_shadow_view.clone()),
			],
			[],
		)?;

		/* shadow pipeline */
		let vertex_input_state = VertexInputState::new()
			.binding(0, super::model::VERTEX_BINDINGS[0].clone())
			.attribute(0, super::model::VERTEX_ATTRIBUTES[0]);
		let rasterization_state = RasterizationState {
			depth_bias: Some(DepthBiasState::default()),
			..Default::default()
		};
		let pipeline_layout_info = PipelineLayoutCreateInfo {
			push_constant_ranges: vec![PushConstantRange {
				stages: ShaderStages::VERTEX,
				offset: 0,
				size: std::mem::size_of::<Mat4>().try_into().unwrap(),
			}],
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_info)?;
		let rendering_formats = PipelineRenderingCreateInfo {
			depth_attachment_format: Some(Format::D16_UNORM),
			..Default::default()
		};
		let depth_stencil_state = DepthStencilState {
			depth: Some(DepthState {
				compare_op: CompareOp::Less,
				write_enable: true,
			}),
			..Default::default()
		};
		let vs_entry_point = vs_shadow::load(device.clone())?.entry_point("main").unwrap();
		let pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![PipelineShaderStageCreateInfo::new(vs_entry_point)],
			vertex_input_state: Some(vertex_input_state),
			input_assembly_state: Some(Default::default()),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(rasterization_state),
			multisample_state: Some(Default::default()),
			depth_stencil_state: Some(depth_stencil_state),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(rendering_formats.into()),
			..GraphicsPipelineCreateInfo::layout(pipeline_layout)
		};
		let shadow_pipeline = GraphicsPipeline::new(device.clone(), None, pipeline_info)?;

		let dir_light_cb = Vec::with_capacity(dir_light_shadow_img.array_layers().try_into().unwrap());

		Ok(Self {
			dir_light_projviews: Default::default(),
			dir_light_buf,
			dir_light_shadow: dir_light_shadow_view,
			dir_light_shadow_layers,
			dir_light_cb: Mutex::new(dir_light_cb),
			all_lights_set,
			shadow_pipeline,
			update_needed: None,
		})
	}

	pub fn update_dir_light(&mut self, light: &DirectionalLight, transform: &Transform, cut_camera_frustums: [DMat4; 3])
	{
		let direction = transform.rotation_quat() * DVec3::NEG_Z;

		// Fit the light view and projection matrices to different sections of the camera frustum.
		// Most of this is adapted from here: https://learnopengl.com/Guest-Articles/2021/CSM
		for (i, cut_frustum) in cut_camera_frustums.iter().enumerate() {
			let camera_projview_inv = cut_frustum.inverse();
			let mut frustum_corners: [DVec4; 8] = Default::default();
			let mut corner_i = 0;
			for x in 0..2 {
				for y in 0..2 {
					for z in 0..2 {
						let pt_x = 2 * x - 1;
						let pt_y = 2 * y - 1;
						let pt = camera_projview_inv * IVec4::new(pt_x, pt_y, z, 1).as_dvec4();
						frustum_corners[corner_i] = pt / pt.w;
						corner_i += 1;
					}
				}
			}

			let center = frustum_corners.iter().sum::<DVec4>() * (1.0 / 8.0);
			let view = DMat4::look_to_lh(center.truncate(), direction, DVec3::Y);

			let mut min_x = f64::MAX;
			let mut max_x = f64::MIN;
			let mut min_y = f64::MAX;
			let mut max_y = f64::MIN;
			let mut min_z = f64::MAX;
			let mut max_z = f64::MIN;
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

			let proj = DMat4::orthographic_lh(min_x, max_x, min_y, max_y, min_z, max_z);

			self.dir_light_projviews[i] = proj * view;
		}

		let mut projviews_f32: [Mat4; 3] = Default::default();
		for (i, projview) in self.dir_light_projviews.iter().enumerate() {
			projviews_f32[i] = projview.as_mat4();
		}
		let dir_light_data = DirLightData {
			projviews: projviews_f32,
			direction: direction.as_vec3().extend(0.0),
			color_intensity: light.color.extend(light.intensity),
		};
		self.update_needed = Some(dir_light_data);
	}

	pub fn add_dir_light_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		let mut dir_light_cb = self.dir_light_cb.lock().unwrap();
		if dir_light_cb.len() == dir_light_cb.capacity() {
			panic!("attempted to add too many command buffers for directional light rendering");
		}
		dir_light_cb.push(cb);
	}

	pub fn execute_shadow_rendering(
		&mut self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
	) -> crate::Result<()>
	{
		if let Some(dir_light_data) = self.update_needed.take() {
			cb_builder.update_buffer(self.dir_light_buf.clone(), Box::from([dir_light_data]))?;
		}

		let mut dir_light_cb = self.dir_light_cb.lock().unwrap();
		let shadow_iter = dir_light_cb.drain(..).zip(self.dir_light_shadow_layers.iter());
		for (shadow_cb, shadow_layer_image_view) in shadow_iter {
			let shadow_render_info = RenderingInfo {
				depth_attachment: Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Depth(1.0)),
					..RenderingAttachmentInfo::image_view(shadow_layer_image_view.clone())
				}),
				contents: SubpassContents::SecondaryCommandBuffers,
				..Default::default()
			};
			cb_builder
				.begin_rendering(shadow_render_info)?
				.execute_commands(shadow_cb)?
				.end_rendering()?;
		}

		Ok(())
	}

	pub fn get_dir_light_shadow(&self) -> &Arc<ImageView>
	{
		&self.dir_light_shadow
	}
	pub fn get_dir_light_projviews(&self) -> [DMat4; 3]
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
