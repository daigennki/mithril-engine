/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::{Arc, Mutex};
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState},
	depth_stencil::CompareOp,
	input_assembly::PrimitiveTopology,
	viewport::Viewport,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sampler::Sampler;

use crate::GenericEngineError;

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
			.begin_render_pass(comp_rp_info, SubpassContents::Inline)?
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

/// A renderer that implements Moment-Based Order-Independent Transparency (MBOIT).
pub struct MomentTransparencyRenderer
{
	moments_fb: Arc<Framebuffer>,
	moments_pl: super::pipeline::Pipeline,
	transparency_compositing_pl: super::pipeline::Pipeline,

	stage3_inputs: Arc<PersistentDescriptorSet>,
	stage4_inputs: Arc<PersistentDescriptorSet>,

	transparency_moments_cb: Mutex<Option<SecondaryAutoCommandBuffer>>,
	transparency_cb: Mutex<Option<SecondaryAutoCommandBuffer>>,
}
impl MomentTransparencyRenderer
{
	pub fn new(
		memory_allocator: &StandardMemoryAllocator,
		descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
		color_image: Arc<AttachmentImage>,
		depth_image: Arc<AttachmentImage>,
		main_sampler: Arc<Sampler>,
	) -> Result<Self, GenericEngineError>
	{
		let vk_dev = memory_allocator.device().clone();

		let moments_rp = vulkano::ordered_passes_renderpass!(vk_dev.clone(),
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
		)?;

		let mut moments_blend = ColorBlendState::new(3).blend_additive();
		moments_blend.attachments[0].blend.as_mut().unwrap().alpha_op = BlendOp::Add;
		moments_blend.attachments[2].blend = Some(AttachmentBlend {
			color_op: BlendOp::Min,
			color_source: BlendFactor::One,
			color_destination: BlendFactor::One,
			..AttachmentBlend::ignore_source()
		});
		let moments_pl = super::pipeline::Pipeline::new(
			PrimitiveTopology::TriangleList,
			"basic_3d_nonorm.vert.spv".into(),
			Some(("mboit_moments.frag.spv".into(), moments_blend)),
			None,
			vec![(2, 0, main_sampler)],
			Subpass::from(moments_rp.clone(), 0).unwrap(),
			CompareOp::Less,
			false,
			descriptor_set_allocator.clone(),
		)?;

		let wboit_compositing_blend = ColorBlendState::new(1).blend_alpha();
		let transparency_compositing_pl = super::pipeline::Pipeline::new(
			PrimitiveTopology::TriangleList,
			"fill_viewport_posonly.vert.spv".into(),
			Some(("mboit_compositing.frag.spv".into(), wboit_compositing_blend)),
			None,
			vec![],
			Subpass::from(moments_rp.clone(), 2).unwrap(),
			CompareOp::Always,
			false,
			descriptor_set_allocator,
		)?;

		let (moments_fb, stage3_inputs, stage4_inputs) = create_mboit_framebuffer(
			memory_allocator,
			moments_rp,
			color_image,
			depth_image,
			&transparency_compositing_pl,
		)?;

		Ok(MomentTransparencyRenderer {
			moments_fb,
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
		memory_allocator: &StandardMemoryAllocator,
		color_image: Arc<AttachmentImage>,
		depth_image: Arc<AttachmentImage>,
	) -> Result<(), GenericEngineError>
	{
		let moments_rp = self.moments_fb.render_pass().clone();
		let (moments_fb, stage3_inputs, stage4_inputs) = create_mboit_framebuffer(
			memory_allocator,
			moments_rp,
			color_image,
			depth_image,
			&self.transparency_compositing_pl,
		)?;
		self.moments_fb = moments_fb;
		self.stage3_inputs = stage3_inputs;
		self.stage4_inputs = stage4_inputs;
		Ok(())
	}

	/// Do the OIT processing using the secondary command buffers that have already been received.
	pub fn process_transparency(
		&self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
	) -> Result<(), GenericEngineError>
	{
		let moments_cb = self.transparency_moments_cb.lock().unwrap().take().unwrap();
		let transparency_cb = self.transparency_cb.lock().unwrap().take().unwrap();

		let rp_info = RenderPassBeginInfo {
			clear_values: vec![
				Some(ClearValue::Float([1.0, 1.0, 1.0, 1.0])), // moments
				Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])), // optical depth
				Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])), // minimum depth
				Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])), // accum
				Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])), // revealage
				None,                                          // color
				None,                                          // depth
			],
			..RenderPassBeginInfo::framebuffer(self.moments_fb.clone())
		};
		let fb_extent = self.moments_fb.extent();
		let viewport = Viewport {
			origin: [0.0, 0.0],
			dimensions: [fb_extent[0] as f32, fb_extent[1] as f32],
			depth_range: 0.0..1.0,
		};

		// draw the objects to the transparency framebuffer, then composite the transparency onto the main framebuffer
		cb.begin_render_pass(rp_info, SubpassContents::SecondaryCommandBuffers)?
			.execute_commands(moments_cb)?
			.next_subpass(SubpassContents::SecondaryCommandBuffers)?
			.execute_commands(transparency_cb)?
			.next_subpass(SubpassContents::Inline)?
			.set_viewport(0, [viewport]);
		self.transparency_compositing_pl.bind(cb);
		super::bind_descriptor_set(cb, 3, vec![self.stage4_inputs.clone()])?;
		cb.draw(3, 1, 0, 0)?.end_render_pass()?;
		Ok(())
	}

	pub fn add_transparency_moments_cb(&self, command_buffer: SecondaryAutoCommandBuffer)
	{
		*self.transparency_moments_cb.lock().unwrap() = Some(command_buffer)
	}
	pub fn add_transparency_cb(&self, command_buffer: SecondaryAutoCommandBuffer)
	{
		*self.transparency_cb.lock().unwrap() = Some(command_buffer)
	}

	pub fn get_moments_pipeline(&self) -> &super::pipeline::Pipeline
	{
		&self.moments_pl
	}

	pub fn get_stage3_inputs(&self) -> Arc<PersistentDescriptorSet>
	{
		self.stage3_inputs.clone()
	}

	pub fn framebuffer(&self) -> Arc<Framebuffer>
	{
		self.moments_fb.clone()
	}
}

fn create_mboit_framebuffer(
	memory_allocator: &StandardMemoryAllocator,
	render_pass: Arc<RenderPass>,
	color_img: Arc<AttachmentImage>,
	depth_img: Arc<AttachmentImage>,
	stage4_pl: &super::pipeline::Pipeline,
) -> Result<(Arc<Framebuffer>, Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>), GenericEngineError>
{
	let extent = depth_img.dimensions().width_height();
	let oit_images_usage = ImageUsage {
		transient_attachment: true,
		input_attachment: true,
		..Default::default()
	};
	let moments_img = AttachmentImage::with_usage(memory_allocator, extent, Format::R32G32B32A32_SFLOAT, oit_images_usage)?;
	let od_img = AttachmentImage::with_usage(memory_allocator, extent, Format::R32_SFLOAT, oit_images_usage)?;
	let min_depth_img = AttachmentImage::with_usage(memory_allocator, extent, Format::R32_SFLOAT, oit_images_usage)?;
	let accum = AttachmentImage::with_usage(memory_allocator, extent, Format::R16G16B16A16_SFLOAT, oit_images_usage)?;
	let revealage = AttachmentImage::with_usage(memory_allocator, extent, Format::R8_UNORM, oit_images_usage)?;

	let moments_view = ImageView::new_default(moments_img)?;
	let od_view = ImageView::new_default(od_img)?;
	let min_depth_view = ImageView::new_default(min_depth_img)?;
	let accum_view = ImageView::new_default(accum)?;
	let revealage_view = ImageView::new_default(revealage)?;

	let fb_create_info = FramebufferCreateInfo {
		attachments: vec![
			moments_view.clone(),
			od_view.clone(),
			min_depth_view.clone(),
			accum_view.clone(),
			revealage_view.clone(),
			ImageView::new_default(color_img)?,
			ImageView::new_default(depth_img)?,
		],
		..Default::default()
	};

	let stage3_inputs = stage4_pl.new_descriptor_set(
		3,
		[
			WriteDescriptorSet::image_view(0, moments_view),
			WriteDescriptorSet::image_view(1, od_view),
			WriteDescriptorSet::image_view(2, min_depth_view.clone()),
		],
	)?;
	let stage4_inputs = stage4_pl.new_descriptor_set(
		3,
		[
			WriteDescriptorSet::image_view(0, accum_view),
			WriteDescriptorSet::image_view(1, revealage_view),
			WriteDescriptorSet::image_view(2, min_depth_view),
		],
	)?;

	Ok((
		Framebuffer::new(render_pass.clone(), fb_create_info)?,
		stage3_inputs,
		stage4_inputs,
	))
}

/*fn create_transparency_framebuffer(
	memory_allocator: &StandardMemoryAllocator,
	depth_img: Arc<AttachmentImage>, render_pass: Arc<RenderPass>, pipeline: &super::pipeline::Pipeline,
	attachment1_format: Format, attachment2_format: Format,
) -> Result<(Arc<Framebuffer>, Arc<PersistentDescriptorSet>), GenericEngineError>
{
	let usage = ImageUsage { sampled: true, ..Default::default() };

	let extent = depth_img.dimensions().width_height();
	let attachment1 = AttachmentImage::with_usage(memory_allocator, extent, attachment1_format, usage)?;
	let attachment2 = AttachmentImage::with_usage(memory_allocator, extent, attachment2_format, usage)?;
	let fb_create_info = FramebufferCreateInfo {
		attachments: vec![
			ImageView::new_default(attachment1)?,
			ImageView::new_default(attachment2)?,
			ImageView::new_default(depth_img.clone())?,
		],
		..Default::default()
	};

	Ok(Framebuffer::new(render_pass.clone(), fb_create_info)?)
}*/
