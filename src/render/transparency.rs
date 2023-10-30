/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use std::sync::{Arc, Mutex};
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer, SubpassContents, 
	RenderingAttachmentInfo, RenderingInfo,
};
use vulkano::descriptor_set::{
	allocator::StandardDescriptorSetAllocator, 
	layout::{DescriptorSetLayout, DescriptorSetLayoutCreateInfo, DescriptorSetLayoutBinding, DescriptorType},
	DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet
};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{sampler::Sampler, view::ImageView, Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::{layout::PushConstantRange, Pipeline, PipelineBindPoint};
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState, ColorBlendAttachmentState},
	depth_stencil::{CompareOp, DepthState, DepthStencilState},
	input_assembly::PrimitiveTopology,
	viewport::Viewport,
	subpass::PipelineRenderingCreateInfo,
	GraphicsPipeline,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

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

mod vs_nonorm {
	vulkano_shaders::shader! {
		ty: "vertex",
		bytes: "shaders/basic_3d_nonorm.vert.spv",
	}
}
mod fs_moments {
	vulkano_shaders::shader! {
		ty: "fragment",
		bytes: "shaders/mboit_moments.frag.spv",
	}
}
mod vs_fill_viewport {
	vulkano_shaders::shader! {
		ty: "fragment",
		bytes: "shaders/fill_viewport.vert.spv",
	}
}
mod fs_oit_compositing {
	vulkano_shaders::shader! {
		ty: "fragment",
		bytes: "shaders/wboit_compositing.frag.spv"
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
		dimensions: [u32; 2],
		main_sampler: Arc<Sampler>,
	) -> Result<Self, GenericEngineError>
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
		let base_color_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, DescriptorSetLayoutBinding { // binding 0: sampler0
					stages: ShaderStages::FRAGMENT,
					immutable_samplers: vec![ main_sampler ],
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
				}),
				(1, DescriptorSetLayoutBinding { // binding 1: base_color
					stages: ShaderStages::FRAGMENT,
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
				})
			].into(),
			..Default::default()
		};
		let base_color_set_layout = DescriptorSetLayout::new(device.clone(), base_color_set_layout_info)?;

		let mut moments_blend = ColorBlendState::with_attachment_states(3, ColorBlendAttachmentState { 
			blend: Some(AttachmentBlend::additive()),
			..Default::default()
		});
		moments_blend.attachments[0].blend.as_mut().unwrap().alpha_blend_op = BlendOp::Add;
		moments_blend.attachments[2].blend = Some(AttachmentBlend {
			color_blend_op: BlendOp::Min,
			src_color_blend_factor: BlendFactor::One,
			dst_color_blend_factor: BlendFactor::One,
			..AttachmentBlend::ignore_source()
		});

		let moments_depth_stencil_state = DepthStencilState {
			depth: Some(DepthState {
				write_enable: false,
				compare_op: CompareOp::Less,
			}),
			..Default::default()
		};

		let moments_rendering = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![ 
				Some(Format::R32G32B32A32_SFLOAT), // moments
				Some(Format::R32_SFLOAT), // optical_depth
				Some(Format::R32_SFLOAT), // min_depth
			],
			depth_attachment_format: Some(super::MAIN_DEPTH_FORMAT),
			..Default::default()
		};
		let moments_pl = super::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleList,
			&[ vs_nonorm::load(device.clone())?, fs_moments::load(device.clone())? ],
			Some(moments_blend),
			vec![ base_color_set_layout ],
			vec![ 
				PushConstantRange { // push constant for projview matrix
					stages: ShaderStages::VERTEX,
					offset: 0,
					size: (std::mem::size_of::<Mat4>() * 2).try_into().unwrap(),
				}
			],
			moments_rendering,
			Some(moments_depth_stencil_state),
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
		let stage3_inputs_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, input_binding.clone()), // moments
				(1, input_binding.clone()), // optical_depth
				(2, input_binding.clone()), // min_depth
			].into(),
			..Default::default()
		};
		let stage3_inputs_layout = DescriptorSetLayout::new(device.clone(), stage3_inputs_layout_info)?;

		//
		/* Stage 4: Composite transparency image onto opaque image */
		//
		let stage4_inputs_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, input_binding.clone()), // accum
				(1, input_binding), // revealage
			].into(),
			..Default::default()
		};
		let stage4_inputs_layout = DescriptorSetLayout::new(device.clone(), stage4_inputs_layout_info)?;

		let wboit_compositing_blend = ColorBlendState::with_attachment_states(1, ColorBlendAttachmentState {
			blend: Some(AttachmentBlend::alpha()),
			..Default::default()
		});

		let compositing_rendering = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![ Some(Format::R16G16B16A16_SFLOAT) ],
			..Default::default()
		};
		let transparency_compositing_pl = super::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleList,
			&[ vs_fill_viewport::load(device.clone())?, fs_oit_compositing::load(device.clone())? ],
			Some(wboit_compositing_blend),
			vec![ stage4_inputs_layout.clone() ],
			vec![],
			compositing_rendering,
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
	) -> Result<(), GenericEngineError>
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
	) -> Result<(), GenericEngineError>
	{
		let extent3 = self.images.moments.image().extent();
		let img_extent = [ extent3[0], extent3[1] ];
		
		let stage2_rendering_info = RenderingInfo {
			render_area_extent: img_extent,
			color_attachments: vec![
				Some(RenderingAttachmentInfo { // moments
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.moments.clone())
				}),
				Some(RenderingAttachmentInfo { // optical_depth
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.optical_depth.clone())
				}),
				Some(RenderingAttachmentInfo { // min_depth
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
				Some(RenderingAttachmentInfo { // accum
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.accum.clone())
				}),
				Some(RenderingAttachmentInfo { // revealage
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
			color_attachments: vec![ //[color]
				Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Load,
					store_op: AttachmentStoreOp::Store,
					..RenderingAttachmentInfo::image_view(color_image.clone())
				}),
			],
			contents: SubpassContents::Inline,
			..Default::default()
		};

		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [img_extent[0] as f32, img_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};

		let moments_cb = Arc::new(self.transparency_moments_cb.lock().unwrap().take().unwrap());
		let transparency_cb = Arc::new(self.transparency_cb.lock().unwrap().take().unwrap());

		// draw the objects to the transparency framebuffer, then composite the transparency onto the main framebuffer
		cb.begin_rendering(stage2_rendering_info)?
			.execute_commands(moments_cb)?
			.end_rendering()?
			.begin_rendering(stage3_rendering_info)?
			.execute_commands(transparency_cb)?
			.end_rendering()?
			.begin_rendering(stage4_rendering_info)?
			.set_viewport(0, [viewport].as_slice().into())?
			.bind_pipeline_graphics(self.transparency_compositing_pl.clone())?
			.bind_descriptor_sets(
				PipelineBindPoint::Graphics, 
				self.transparency_compositing_pl.layout().clone(), 
				0, 
				vec![self.stage4_inputs.clone()]
			)?
			.draw(3, 1, 0, 0)?
			.end_rendering()?;
		Ok(())
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
	pub moments: Arc<ImageView>,
	pub optical_depth: Arc<ImageView>,
	pub min_depth: Arc<ImageView>,
	pub accum: Arc<ImageView>,
	pub revealage: Arc<ImageView>,
}

fn create_mboit_images(
	memory_allocator: Arc<StandardMemoryAllocator>,
	descriptor_set_allocator: &StandardDescriptorSetAllocator,
	extent: [u32; 2],
	stage3_inputs_layout: Arc<DescriptorSetLayout>,
	stage4_inputs_layout: Arc<DescriptorSetLayout>,
) -> Result<(MomentImageBundle, Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>), GenericEngineError>
{

	let mut image_create_info = ImageCreateInfo {
		usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
		format: Format::R32G32B32A32_SFLOAT,
		extent: [ extent[0], extent[1], 1 ],
		..Default::default()
	};
	let moments_img = Image::new(memory_allocator.clone(), image_create_info.clone(), AllocationCreateInfo::default())?;
	image_create_info.format = Format::R32_SFLOAT;
	let od_img = Image::new(memory_allocator.clone(), image_create_info.clone(), AllocationCreateInfo::default())?;
	image_create_info.format = Format::R32_SFLOAT;
	let min_depth_img = Image::new(memory_allocator.clone(), image_create_info.clone(), AllocationCreateInfo::default())?;
	image_create_info.format = Format::R16G16B16A16_SFLOAT;
	let accum = Image::new(memory_allocator.clone(), image_create_info.clone(), AllocationCreateInfo::default())?;
	image_create_info.format = Format::R8_UNORM;
	let revealage = Image::new(memory_allocator, image_create_info, AllocationCreateInfo::default())?;

	let moments_view = ImageView::new_default(moments_img)?;
	let od_view = ImageView::new_default(od_img)?;
	let min_depth_view = ImageView::new_default(min_depth_img)?;
	let accum_view = ImageView::new_default(accum)?;
	let revealage_view = ImageView::new_default(revealage)?;

	let image_bundle = MomentImageBundle {
		moments: moments_view.clone(),
		optical_depth: od_view.clone(),
		min_depth: min_depth_view.clone(),
		accum: accum_view.clone(),
		revealage: revealage_view.clone()
	};

	let stage3_inputs = PersistentDescriptorSet::new(
		descriptor_set_allocator,
		stage3_inputs_layout,
		[
			WriteDescriptorSet::image_view(0, moments_view),
			WriteDescriptorSet::image_view(1, od_view),
			WriteDescriptorSet::image_view(2, min_depth_view),
		],
		[]
	)?;

	let stage4_inputs = PersistentDescriptorSet::new(
		descriptor_set_allocator,
		stage4_inputs_layout,
		[
			WriteDescriptorSet::image_view(0, accum_view),
			WriteDescriptorSet::image_view(1, revealage_view),
		],
		[]
	)?;

	Ok((
		image_bundle,
		stage3_inputs,
		stage4_inputs,
	))
}

