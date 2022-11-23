/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::{Arc, Mutex};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::{
	color_blend::{BlendOp, ColorBlendState}, depth_stencil::CompareOp, input_assembly::PrimitiveTopology, viewport::Viewport,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::sampler::Sampler;

use crate::GenericEngineError;

/// A renderer that implements Weight-Based Order-Independent Transparency (WBOIT).
pub struct TransparencyRenderer
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
}

/// A renderer that implements Moment-Based Order-Independent Transparency (MBOIT).
pub struct MomentTransparencyRenderer
{
	moments_fb: Arc<Framebuffer>,
	/// The descriptor set holding the sampled images and extent buffer for `moments_fb`.
	moments_set: Arc<PersistentDescriptorSet>,
	moments_pl: super::pipeline::Pipeline,

	transparency_fb: Arc<Framebuffer>,

	/// The descriptor set holding the sampled images and extent buffer for `transparency_fb`.
	transparency_set: Arc<PersistentDescriptorSet>,
	transparency_compositing_pl: super::pipeline::Pipeline,
	compositing_rp: Arc<RenderPass>,

	transparency_moments_cb: Mutex<Option<SecondaryAutoCommandBuffer>>,
	transparency_cb: Mutex<Option<SecondaryAutoCommandBuffer>>,
}
impl MomentTransparencyRenderer
{
	pub fn new(
		memory_allocator: &StandardMemoryAllocator, descriptor_set_allocator: &StandardDescriptorSetAllocator,
		depth_image: Arc<AttachmentImage>, main_sampler: Arc<Sampler>
	) -> Result<Self, GenericEngineError>
	{
		let vk_dev = memory_allocator.device().clone();
		
		let moments_rp = vulkano::single_pass_renderpass!(vk_dev.clone(),
			attachments: {
				moments: {
					load: Clear,
					store: Store,
					format: Format::R32G32B32A32_SFLOAT,
					samples: 1,
				},
				optical_depth: {
					load: Clear,
					store: Store,
					format: Format::R32_SFLOAT,
					samples: 1,
				},
				depth: {
					load: Load,
					store: Store,
					format: Format::D16_UNORM,
					samples: 1,
				}
			},
			pass: {
				color: [moments, optical_depth],
				depth_stencil: { depth }
			}
		)?;

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

		let moments_subpass = moments_rp.clone().first_subpass();
		let moments_blend = ColorBlendState::new(2).blend_additive();
		moments_blend.attachments[0].blend.unwrap().alpha_op = BlendOp::Add;
		let moments_pl = super::pipeline::Pipeline::new(
			PrimitiveTopology::TriangleList,
			"basic_3d_nonorm.vert.spv".into(),
			Some(("mboit_moments.frag.spv".into(), moments_blend)),
			None,
			vec![ (2, 0, main_sampler) ],
			moments_subpass,
			CompareOp::Less,
			false,
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

		let (moments_fb, moments_set) = create_transparency_framebuffer(
			memory_allocator,
			descriptor_set_allocator,
			depth_image.clone(),
			moments_rp,
			&transparency_compositing_pl, // not actually the pipeline we'll use the descriptor set with, but it happens to be compatible
			Format::R32G32B32A32_SFLOAT,
			Format::R32_SFLOAT,
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

		Ok(MomentTransparencyRenderer {
			moments_fb,
			moments_set,
			moments_pl,
			transparency_fb,
			transparency_set,
			transparency_compositing_pl,
			compositing_rp,
			transparency_moments_cb: Mutex::new(None),
			transparency_cb: Mutex::new(None),
		})
	}

	/// Resize the output image to match a resized depth image.
	pub fn resize_image(
		&mut self, memory_allocator: &StandardMemoryAllocator, descriptor_set_allocator: &StandardDescriptorSetAllocator,
		depth_image: Arc<AttachmentImage>,
	) -> Result<(), GenericEngineError>
	{
		let moments_rp = self.moments_fb.render_pass().clone();
		let (moments_fb, moments_set) = create_transparency_framebuffer(
			memory_allocator,
			descriptor_set_allocator,
			depth_image.clone(),
			moments_rp,
			&self.transparency_compositing_pl,
			Format::R32G32B32A32_SFLOAT,
			Format::R32_SFLOAT,
		)?;
		self.moments_fb = moments_fb;
		self.moments_set = moments_set;

		let transparency_rp = self.transparency_fb.render_pass().clone();
		let (transparency_fb, transparency_set) = create_transparency_framebuffer(
			memory_allocator,
			descriptor_set_allocator,
			depth_image.clone(),
			transparency_rp,
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
		&self, 
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		framebuffer: Arc<Framebuffer>,
	) -> Result<(), GenericEngineError>
	{
		let moments_cb = self.transparency_moments_cb.lock().unwrap().take().unwrap();
		let transparency_cb = self.transparency_cb.lock().unwrap().take().unwrap();

		let moments_rp_info = RenderPassBeginInfo {
			clear_values: vec![
				Some(ClearValue::Float([1.0, 1.0, 1.0, 1.0])), // accum
				Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])), // revealage
				None,                                          // depth; just load it
			],
			..RenderPassBeginInfo::framebuffer(self.moments_fb.clone())
		};

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
		cb.begin_render_pass(moments_rp_info, SubpassContents::SecondaryCommandBuffers)?
			.execute_commands(moments_cb)?
			.end_render_pass()?
			.begin_render_pass(transparency_rp_info, SubpassContents::SecondaryCommandBuffers)?
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

	pub fn add_transparency_moments_cb(&self, command_buffer: SecondaryAutoCommandBuffer)
	{
		*self.transparency_moments_cb.lock().unwrap() = Some(command_buffer)
	}
	pub fn add_transparency_cb(&self, command_buffer: SecondaryAutoCommandBuffer)
	{
		*self.transparency_cb.lock().unwrap() = Some(command_buffer)
	}

	pub fn get_moments_descriptor_set(&self) -> Arc<PersistentDescriptorSet>
	{
		self.moments_set.clone()
	}

	pub fn get_moments_pipeline(&self) -> &super::pipeline::Pipeline
	{
		&self.moments_pl
	}

	pub fn moments_framebuffer(&self) -> Arc<Framebuffer>
	{
		self.moments_fb.clone()
	}

	pub fn framebuffer(&self) -> Arc<Framebuffer>
	{
		self.transparency_fb.clone()
	}
}

fn create_transparency_framebuffer(
	memory_allocator: &StandardMemoryAllocator, descriptor_set_allocator: &StandardDescriptorSetAllocator,
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
	let buf_usage = BufferUsage { uniform_buffer: true, ..BufferUsage::empty() };
	let descriptor_set = pipeline.new_descriptor_set(descriptor_set_allocator, 3, [
		WriteDescriptorSet::image_view(0, fb_create_info.attachments[0].clone()),
		WriteDescriptorSet::image_view(1, fb_create_info.attachments[1].clone()),
		WriteDescriptorSet::buffer(2, CpuAccessibleBuffer::from_iter(memory_allocator, buf_usage, false, extent)?),
	])?;

	Ok((Framebuffer::new(render_pass.clone(), fb_create_info)?, descriptor_set))
}
