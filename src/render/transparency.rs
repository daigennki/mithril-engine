 /* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::format::Format;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, view::ImageView};
use vulkano::pipeline::graphics::{
	color_blend::ColorBlendState, depth_stencil::CompareOp, input_assembly::PrimitiveTopology, viewport::Viewport
};
use vulkano::device::{Device, DeviceOwned};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents};

use crate::GenericEngineError;


/// A renderer that implements Order-Independent Transparency (OIT).
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
	pub fn new(vk_dev: Arc<Device>, depth_image: Arc<AttachmentImage>) -> Result<Self, GenericEngineError>
	{
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
					format: Format::B8G8R8A8_SRGB,
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
			Some("wboit_compositing.frag.spv".into()),
			None,
			vec![],
			wboit_compositing_subpass,
			None,
			CompareOp::Always,
			Some(wboit_compositing_blend),
			false,
			None
		)?;

		let (transparency_fb, transparency_set) = create_transparency_framebuffer(
			depth_image.clone(), transparency_rp, &transparency_compositing_pl
		)?;

		Ok(TransparencyRenderer {
			transparency_fb,
			transparency_set,
			transparency_compositing_pl,
			compositing_rp,
		})
	}

	/// Resize the output image to match a resized depth image.
	pub fn resize_image(&mut self, depth_image: Arc<AttachmentImage>) -> Result<(), GenericEngineError>
	{
		let render_pass = self.transparency_fb.render_pass().clone();
		let (transparency_fb, transparency_set) = create_transparency_framebuffer(
			depth_image.clone(), render_pass, &self.transparency_compositing_pl
		)?;
		self.transparency_fb = transparency_fb;
		self.transparency_set = transparency_set;
		Ok(())
	}

	pub fn composite_transparency(
		&self, cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, framebuffer: Arc<Framebuffer>
	) -> Result<(), GenericEngineError>
	{
		let mut rp_begin_info = RenderPassBeginInfo::framebuffer(framebuffer.clone());
		rp_begin_info.render_pass = self.compositing_rp.clone();
		rp_begin_info.clear_values = vec![None, None];

		let fb_extent = framebuffer.extent();
		let viewport = Viewport {
			origin: [0.0, 0.0],
			dimensions: [ fb_extent[0] as f32, fb_extent[1] as f32 ],
			depth_range: 0.0..1.0,
		};

		cb.begin_render_pass(rp_begin_info, SubpassContents::Inline)?;
		cb.set_viewport(0, [viewport]);
		self.transparency_compositing_pl.bind(cb);
		super::bind_descriptor_set(cb, 0, vec![ self.transparency_set.clone() ])?;
		cb.draw(3, 1, 0, 0)?;
		cb.end_render_pass()?;
		Ok(())
	}

	pub fn framebuffer(&self) -> Arc<Framebuffer>
	{
		self.transparency_fb.clone()
	}
}
fn create_transparency_framebuffer(
	depth_img: Arc<AttachmentImage>, render_pass: Arc<RenderPass>, pipeline: &super::pipeline::Pipeline
) -> Result<(Arc<Framebuffer>, Arc<PersistentDescriptorSet>), GenericEngineError>
{
	let usage = ImageUsage{ sampled: true, ..Default::default() };
	let vk_dev = render_pass.device().clone();

	let extent = depth_img.dimensions().width_height();
	let accum = AttachmentImage::with_usage(vk_dev.clone(), extent, Format::R16G16B16A16_SFLOAT, usage)?;
	let revealage = AttachmentImage::with_usage(vk_dev.clone(), extent, Format::R8_UNORM, usage)?;
	let fb_create_info = FramebufferCreateInfo {
		attachments: vec![
			ImageView::new_default(accum)?,
			ImageView::new_default(revealage)?,
			ImageView::new_default(depth_img.clone())?,
		],
		..Default::default()
	};
	let buf_usage = BufferUsage { uniform_buffer: true, ..BufferUsage::empty() };
	let descriptor_set = pipeline.new_descriptor_set(0, [
		WriteDescriptorSet::image_view(0, fb_create_info.attachments[0].clone()),
		WriteDescriptorSet::image_view(1, fb_create_info.attachments[1].clone()),
		WriteDescriptorSet::buffer(2, CpuAccessibleBuffer::from_iter(vk_dev.clone(), buf_usage, false, extent)?)
	])?;

	Ok((Framebuffer::new(render_pass.clone(), fb_create_info)?, descriptor_set))
}

