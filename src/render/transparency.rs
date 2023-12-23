/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use std::sync::{Arc, Mutex};
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo, SecondaryAutoCommandBuffer,
	SubpassContents,
};
use vulkano::descriptor_set::{
	allocator::StandardDescriptorSetAllocator,
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{
	sampler::{Sampler, SamplerCreateInfo},
	view::ImageView,
	Image, ImageCreateInfo, ImageUsage,
};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, BlendFactor, BlendOp},
	depth_stencil::{CompareOp, DepthState},
	input_assembly::PrimitiveTopology,
	rasterization::{CullMode, RasterizationState},
	viewport::Viewport,
	GraphicsPipeline,
};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

use crate::EngineError;

/// A renderer that implements Weight-Based Order-Independent Transparency (WBOIT).
/*pub struct TransparencyRenderer
{
	transparency_fb: Arc<Framebuffer>,

	/// The descriptor set holding the sampled images and extent buffer for `transparency_fb`.
	transparency_set: Arc<PersistentDescriptorSet>,

	transparency_compositing_pl: super::pipeline::Pipeline,
	compositing_rp: Arc<RenderPass>,
}
impl TransparencyRenderer
{
	pub fn new(
		memory_allocator: &StandardMemoryAllocator, descriptor_set_allocator: &StandardDescriptorSetAllocator,
		depth_image: Arc<AttachmentImage>
	) -> Result<Self, GenericEngineError>
	{
		let vk_dev = memory_allocator.device().clone();
		let transparency_rp = vulkano::single_pass_renderpass!(vk_dev.clone(),
			attachments: {
				accum: {
					load: Clear,
					store: Store,
					format: Format::R16G16B16A16_SFLOAT,
					samples: 1,
				},
				revealage: {
					load: Clear,
					store: Store,
					format: Format::R8_UNORM,
					samples: 1,
				},
				depth: {
					load: Load,
					store: DontCare,
					format: Format::D16_UNORM,
					samples: 1,
				}
			},
			pass: {
				color: [accum, revealage],
				depth_stencil: { depth }
			}
		)?;

		let compositing_rp = vulkano::single_pass_renderpass!(
			vk_dev.clone(),
			attachments: {
				color: {
					load: Load,
					store: Store,
					format: Format::R16G16B16A16_SFLOAT,
					samples: 1,
				},
				depth: {
					load: DontCare,
					store: DontCare,
					format: Format::D16_UNORM,
					samples: 1,
				}
			},
			pass: {
				color: [color],
				depth_stencil: {depth}
			}
		)?;

		let wboit_compositing_subpass = compositing_rp.clone().first_subpass();
		let wboit_compositing_blend = ColorBlendState::new(1).blend_alpha();
		let transparency_compositing_pl = super::pipeline::Pipeline::new(
			PrimitiveTopology::TriangleList,
			"fill_viewport.vert.spv".into(),
			Some(("wboit_compositing.frag.spv".into(), wboit_compositing_blend)),
			None,
			vec![],
			wboit_compositing_subpass,
			CompareOp::Always,
			false,
		)?;

		let (transparency_fb, transparency_set) = create_transparency_framebuffer(
			memory_allocator,
			descriptor_set_allocator,
			depth_image.clone(),
			transparency_rp,
			&transparency_compositing_pl,
			Format::R16G16B16A16_SFLOAT,
			Format::R8_UNORM,
		)?;

		Ok(TransparencyRenderer {
			transparency_fb,
			transparency_set,
			transparency_compositing_pl,
			compositing_rp,
		})
	}

	/// Resize the output image to match a resized depth image.
	pub fn resize_image(
		&mut self, memory_allocator: &StandardMemoryAllocator, descriptor_set_allocator: &StandardDescriptorSetAllocator,
		depth_image: Arc<AttachmentImage>,
	) -> Result<(), GenericEngineError>
	{
		let render_pass = self.transparency_fb.render_pass().clone();
		let (transparency_fb, transparency_set) = create_transparency_framebuffer(
			memory_allocator,
			descriptor_set_allocator,
			depth_image.clone(),
			render_pass,
			&self.transparency_compositing_pl,
			Format::R16G16B16A16_SFLOAT,
			Format::R8_UNORM,
		)?;
		self.transparency_fb = transparency_fb;
		self.transparency_set = transparency_set;
		Ok(())
	}

	/// Composite the drawn transparent objects from the secondary command buffer onto the final framebuffer.
	pub fn process_transparency(
		&self, transparency_cb: SecondaryAutoCommandBuffer, cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		framebuffer: Arc<Framebuffer>,
	) -> Result<(), GenericEngineError>
	{
		let transparency_rp_info = RenderPassBeginInfo {
			clear_values: vec![
				Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])), // accum
				Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])), // revealage
				None,                                          // depth; just load it
			],
			..RenderPassBeginInfo::framebuffer(self.transparency_fb.clone())
		};

		let comp_rp_info = RenderPassBeginInfo {
			render_pass: self.compositing_rp.clone(),
			clear_values: vec![None, None],
			..RenderPassBeginInfo::framebuffer(framebuffer.clone())
		};

		let fb_extent = framebuffer.extent();
		let viewport = Viewport {
			origin: [0.0, 0.0],
			dimensions: [
				fb_extent[0] as f32, fb_extent[1] as f32,
			],
			depth_range: 0.0..1.0,
		};

		// draw the objects to the transparency framebuffer, then composite the transparency onto the main framebuffer
		cb.begin_render_pass(transparency_rp_info, SubpassContents::SecondaryCommandBuffers)?
			.execute_commands(transparency_cb)?
			.end_render_pass()?
	)		.begin_render_pass(comp_rp_info, SubpassContents::Inline)?
			.set_viewport(0, [viewport]);
		self.transparency_compositing_pl.bind(cb);
		super::bind_descriptor_set(cb, 3, vec![self.transparency_set.clone()])?;
		cb.draw(3, 1, 0, 0)?
			.end_render_pass()?;
		Ok(())
	}

	pub fn framebuffer(&self) -> Arc<Framebuffer>
	{
		self.transparency_fb.clone()
	}
}*/

mod vs_nonorm
{
	vulkano_shaders::shader! {
		ty: "vertex",
		path: "src/shaders/basic_3d_nonorm.vert.glsl",
	}
}
mod fs_moments
{
	vulkano_shaders::shader! {
		ty: "fragment",
		path: "src/shaders/mboit_moments.frag.glsl",
	}
}
mod vs_fill_viewport
{
	vulkano_shaders::shader! {
		ty: "vertex",
		path: "src/shaders/fill_viewport.vert.glsl",
	}
}
mod fs_oit_compositing
{
	vulkano_shaders::shader! {
		ty: "fragment",
		path: "src/shaders/wboit_compositing.frag.glsl"
	}
}

/// A renderer that implements Moment-Based Order-Independent Transparency (MBOIT).
///
/// Seems to be a little broken as of 3eb0200 (2023/10/27). (Overlapping and
/// intersecting objects look a little wrong...)
pub struct MomentTransparencyRenderer
{
	images: MomentImageBundle,

	moments_pl: Arc<GraphicsPipeline>,
	transparency_compositing_pl: Arc<GraphicsPipeline>,

	stage3_inputs: Arc<PersistentDescriptorSet>,
	stage4_inputs: Arc<PersistentDescriptorSet>,

	transparency_moments_cb: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,
	transparency_cb: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,
}
impl MomentTransparencyRenderer
{
	pub fn new(
		memory_allocator: Arc<StandardMemoryAllocator>,
		descriptor_set_allocator: &StandardDescriptorSetAllocator,
		mat_tex_set_layout: Arc<DescriptorSetLayout>,
		dimensions: [u32; 2],
	) -> Result<Self, EngineError>
	{
		// The render pass from back when we didn't use dynamic rendering.
		// This is left commented out here so we can get an idea of where each image gets used.
		/*let moments_rp = vulkano::ordered_passes_renderpass!(vk_dev.clone(),
			attachments: {
				moments: {
					load: Clear,
					store: DontCare,
					format: Format::R32G32B32A32_SFLOAT,
					samples: 1,
				},
				optical_depth: {
					load: Clear,
					store: DontCare,
					format: Format::R32_SFLOAT,
					samples: 1,
				},
				min_depth: {
					load: Clear,
					store: DontCare,
					format: Format::R32_SFLOAT,
					samples: 1,
				},
				accum: {
					load: Clear,
					store: DontCare,
					format: Format::R16G16B16A16_SFLOAT,
					samples: 1,
				},
				revealage: {
					load: Clear,
					store: DontCare,
					format: Format::R8_UNORM,
					samples: 1,
				},
				color: {
					load: Load,
					store: Store,
					format: Format::R16G16B16A16_SFLOAT,
					samples: 1,
				},
				depth: {
					load: Load,
					store: Store,
					format: Format::D16_UNORM,
					samples: 1,
				}
			},
			passes: [
				{	// MBOIT stage 2: calculate moments
					color: [moments, optical_depth, min_depth],
					depth_stencil: { depth },
					input: []
				},
				{	// MBOIT stage 3: calculate weights
					color: [accum, revealage],
					depth_stencil: { depth },
					input: [moments, optical_depth, min_depth]
				},
				{	// MBOIT stage 4: composite transparency image onto opaque image
					color: [color],
					depth_stencil: {},
					input: [accum, revealage, min_depth]
				}
			]
		)?;*/

		let device = descriptor_set_allocator.device().clone();

		//
		/* Stage 2: Calculate moments */
		//
		let moments_attachments = [
			(
				Format::R32G32B32A32_SFLOAT, // moments
				Some(AttachmentBlend {
					alpha_blend_op: BlendOp::Add,
					..AttachmentBlend::additive()
				}),
			),
			(
				Format::R32_SFLOAT, // optical_depth
				Some(AttachmentBlend {
					alpha_blend_op: BlendOp::Add,
					..AttachmentBlend::additive()
				}),
			),
			(
				Format::R32_SFLOAT, // min_depth
				Some(AttachmentBlend {
					color_blend_op: BlendOp::Min,
					src_color_blend_factor: BlendFactor::One,
					dst_color_blend_factor: BlendFactor::One,
					..Default::default()
				}),
			),
		];

		let moments_depth_state = DepthState {
			write_enable: false,
			compare_op: CompareOp::Less,
		};

		let moments_pl = super::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleList,
			&[
				vs_nonorm::load(device.clone()).unwrap(),
				fs_moments::load(device.clone()).unwrap(),
			],
			RasterizationState {
				cull_mode: CullMode::Back,
				..Default::default()
			},
			vec![mat_tex_set_layout.clone()],
			&moments_attachments,
			Some((super::MAIN_DEPTH_FORMAT, moments_depth_state)),
			None,
		)?;

		//
		/* Stage 3: Calculate weights */
		//
		// The pipeline for stage 3 depends on the material of each mesh, so they're created outside
		// of this transparency renderer. They'll take the following descriptor set containing images
		// generated in Stage 2.
		let input_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
		};
		let input_sampler = Sampler::new(device.clone(), SamplerCreateInfo::default())
			.map_err(|e| EngineError::vulkan_error("failed to create sampler", e))?;
		let oit_sampler_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			immutable_samplers: vec![input_sampler],
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
		};
		let stage3_inputs_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, oit_sampler_binding.clone()),
				(1, input_binding.clone()), // moments
				(2, input_binding.clone()), // optical_depth
				(3, input_binding.clone()), // min_depth
			]
			.into(),
			..Default::default()
		};
		let stage3_inputs_layout = DescriptorSetLayout::new(device.clone(), stage3_inputs_layout_info)
			.map_err(|e| EngineError::vulkan_error("failed to create descriptor set layout", e))?;

		//
		/* Stage 4: Composite transparency image onto opaque image */
		//
		let stage4_inputs_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, oit_sampler_binding),
				(1, input_binding.clone()), // accum
				(2, input_binding),         // revealage
			]
			.into(),
			..Default::default()
		};
		let stage4_inputs_layout = DescriptorSetLayout::new(device.clone(), stage4_inputs_layout_info)
			.map_err(|e| EngineError::vulkan_error("failed to create descriptor set layout", e))?;

		let transparency_compositing_pl = super::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleList,
			&[
				vs_fill_viewport::load(device.clone()).unwrap(),
				fs_oit_compositing::load(device.clone()).unwrap(),
			],
			RasterizationState::default(),
			vec![stage4_inputs_layout.clone()],
			&[(Format::R16G16B16A16_SFLOAT, Some(AttachmentBlend::alpha()))],
			None,
			None,
		)?;

		/* Create the images and descriptor sets */
		let (images, stage3_inputs, stage4_inputs) = create_mboit_images(
			memory_allocator,
			descriptor_set_allocator,
			dimensions,
			stage3_inputs_layout,
			stage4_inputs_layout,
		)?;

		Ok(MomentTransparencyRenderer {
			images,
			moments_pl,
			transparency_compositing_pl,
			stage3_inputs,
			stage4_inputs,
			transparency_moments_cb: Mutex::new(None),
			transparency_cb: Mutex::new(None),
		})
	}

	/// Resize the output image to match a resized depth image.
	pub fn resize_image(
		&mut self,
		memory_allocator: Arc<StandardMemoryAllocator>,
		descriptor_set_allocator: &StandardDescriptorSetAllocator,
		dimensions: [u32; 2],
	) -> Result<(), EngineError>
	{
		let (moments_images, stage3_inputs, stage4_inputs) = create_mboit_images(
			memory_allocator,
			descriptor_set_allocator,
			dimensions,
			self.stage3_inputs.layout().clone(),
			self.stage4_inputs.layout().clone(),
		)?;

		self.images = moments_images;
		self.stage3_inputs = stage3_inputs;
		self.stage4_inputs = stage4_inputs;
		Ok(())
	}

	/// Do the OIT processing using the secondary command buffers that have already been received.
	pub fn process_transparency(
		&self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		color_image: Arc<ImageView>,
		depth_image: Arc<ImageView>,
	)
	{
		let extent3 = self.images.moments.image().extent();
		let img_extent = [extent3[0], extent3[1]];

		let moments_cb;
		if let Some(m_cb) = self.transparency_moments_cb.lock().unwrap().take() {
			moments_cb = Arc::new(m_cb);
		} else {
			// Skip OIT processing if no transparent materials are in view
			return;
		}
		let transparency_cb = Arc::new(self.transparency_cb.lock().unwrap().take().unwrap());

		let stage2_rendering_info = RenderingInfo {
			render_area_extent: img_extent,
			color_attachments: vec![
				Some(RenderingAttachmentInfo {
					// moments
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.moments.clone())
				}),
				Some(RenderingAttachmentInfo {
					// optical_depth
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.optical_depth.clone())
				}),
				Some(RenderingAttachmentInfo {
					// min_depth
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.min_depth.clone())
				}),
			],
			depth_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(depth_image.clone())
			}),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		};

		let stage3_rendering_info = RenderingInfo {
			render_area_extent: img_extent,
			color_attachments: vec![
				Some(RenderingAttachmentInfo {
					// accum
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.accum.clone())
				}),
				Some(RenderingAttachmentInfo {
					// revealage
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.revealage.clone())
				}),
			],
			depth_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::DontCare,
				..RenderingAttachmentInfo::image_view(depth_image.clone())
			}),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		};

		let stage4_rendering_info = RenderingInfo {
			render_area_extent: img_extent,
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(color_image.clone())
			})],
			contents: SubpassContents::Inline,
			..Default::default()
		};

		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [img_extent[0] as f32, img_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};

		// draw the objects to the transparency framebuffer, then composite the transparency onto the main framebuffer
		cb.begin_rendering(stage2_rendering_info)
			.unwrap()
			.execute_commands(moments_cb)
			.unwrap()
			.end_rendering()
			.unwrap()
			.begin_rendering(stage3_rendering_info)
			.unwrap()
			.execute_commands(transparency_cb)
			.unwrap()
			.end_rendering()
			.unwrap()
			.begin_rendering(stage4_rendering_info)
			.unwrap()
			.set_viewport(0, [viewport].as_slice().into())
			.unwrap()
			.bind_pipeline_graphics(self.transparency_compositing_pl.clone())
			.unwrap()
			.bind_descriptor_sets(
				PipelineBindPoint::Graphics,
				self.transparency_compositing_pl.layout().clone(),
				0,
				vec![self.stage4_inputs.clone()],
			)
			.unwrap()
			.draw(3, 1, 0, 0)
			.unwrap()
			.end_rendering()
			.unwrap();
	}

	pub fn add_transparency_moments_cb(&self, command_buffer: Arc<SecondaryAutoCommandBuffer>)
	{
		*self.transparency_moments_cb.lock().unwrap() = Some(command_buffer)
	}
	pub fn add_transparency_cb(&self, command_buffer: Arc<SecondaryAutoCommandBuffer>)
	{
		*self.transparency_cb.lock().unwrap() = Some(command_buffer)
	}

	pub fn get_moments_pipeline(&self) -> &Arc<GraphicsPipeline>
	{
		&self.moments_pl
	}

	pub fn get_stage3_inputs(&self) -> &Arc<PersistentDescriptorSet>
	{
		&self.stage3_inputs
	}
}

struct MomentImageBundle
{
	moments: Arc<ImageView>,
	optical_depth: Arc<ImageView>,
	min_depth: Arc<ImageView>,
	accum: Arc<ImageView>,
	revealage: Arc<ImageView>,
}

fn create_mboit_images(
	memory_allocator: Arc<StandardMemoryAllocator>,
	descriptor_set_allocator: &StandardDescriptorSetAllocator,
	extent: [u32; 2],
	stage3_inputs_layout: Arc<DescriptorSetLayout>,
	stage4_inputs_layout: Arc<DescriptorSetLayout>,
) -> Result<(MomentImageBundle, Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>), EngineError>
{
	let image_create_info = ImageCreateInfo {
		usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
		format: Format::R32G32B32A32_SFLOAT,
		extent: [extent[0], extent[1], 1],
		..Default::default()
	};
	let image_create_infos = [
		// moments
		image_create_info.clone(),
		// optical_depth
		ImageCreateInfo {
			format: Format::R32_SFLOAT,
			..image_create_info.clone()
		},
		// min_depth
		ImageCreateInfo {
			format: Format::R32_SFLOAT,
			..image_create_info.clone()
		},
		// accum
		ImageCreateInfo {
			format: Format::R16G16B16A16_SFLOAT,
			..image_create_info.clone()
		},
		// revealage
		ImageCreateInfo {
			format: Format::R8_UNORM,
			..image_create_info
		},
	];
	let mut views = Vec::with_capacity(5);
	for info in image_create_infos {
		let new_image = Image::new(memory_allocator.clone(), info.clone(), AllocationCreateInfo::default())
			.map_err(|e| EngineError::vulkan_error("failed to create image", e))?;
		let image_view =
			ImageView::new_default(new_image).map_err(|e| EngineError::vulkan_error("failed to create image view", e))?;
		views.push(image_view);
	}

	let image_bundle = MomentImageBundle {
		moments: views[0].clone(),
		optical_depth: views[1].clone(),
		min_depth: views[2].clone(),
		accum: views[3].clone(),
		revealage: views[4].clone(),
	};

	let stage3_inputs = PersistentDescriptorSet::new(
		descriptor_set_allocator,
		stage3_inputs_layout,
		[
			WriteDescriptorSet::image_view(1, views[0].clone()),
			WriteDescriptorSet::image_view(2, views[1].clone()),
			WriteDescriptorSet::image_view(3, views[2].clone()),
		],
		[],
	)
	.map_err(|e| EngineError::vulkan_error("failed to create descriptor set", e))?;

	let stage4_inputs = PersistentDescriptorSet::new(
		descriptor_set_allocator,
		stage4_inputs_layout,
		[
			WriteDescriptorSet::image_view(1, views[3].clone()),
			WriteDescriptorSet::image_view(2, views[4].clone()),
		],
		[],
	)
	.map_err(|e| EngineError::vulkan_error("failed to create descriptor set", e))?;

	Ok((image_bundle, stage3_inputs, stage4_inputs))
}
