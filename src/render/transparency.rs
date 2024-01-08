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
use vulkano::image::{view::ImageView, Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, BlendOp, ColorBlendAttachmentState, ColorBlendState},
	depth_stencil::{CompareOp, DepthState, DepthStencilState, StencilOpState, StencilOps, StencilState},
	input_assembly::PrimitiveTopology,
	rasterization::{CullMode, RasterizationState},
	subpass::PipelineRenderingCreateInfo,
	viewport::Viewport,
	GraphicsPipeline,
};
use vulkano::pipeline::{
	layout::{PipelineLayoutCreateInfo, PushConstantRange},
	PipelineLayout,
};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

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
		depth_stencil_format: Format,
	) -> crate::Result<Self>
	{
		let device = descriptor_set_allocator.device().clone();

		//
		/* Stage 2: Calculate moments */
		//
		let moments_color_blend_state = ColorBlendState {
			attachments: vec![
				ColorBlendAttachmentState {
					// moments
					blend: Some(AttachmentBlend {
						alpha_blend_op: BlendOp::Add,
						..AttachmentBlend::additive()
					}),
					..Default::default()
				},
				ColorBlendAttachmentState {
					// optical_depth
					blend: Some(AttachmentBlend {
						alpha_blend_op: BlendOp::Add,
						..AttachmentBlend::additive()
					}),
					..Default::default()
				},
				/*ColorBlendAttachmentState {
					// min_depth
					blend: Some(AttachmentBlend {
						color_blend_op: BlendOp::Min,
						src_color_blend_factor: BlendFactor::One,
						dst_color_blend_factor: BlendFactor::One,
						..Default::default()
					}),
					..Default::default()
				},*/
			],
			..Default::default()
		};

		let moments_depth_stencil_state = DepthStencilState {
			depth: Some(DepthState {
				write_enable: false,
				compare_op: CompareOp::Less,
			}),
			..Default::default()
		};

		let stage2_push_constant_size = std::mem::size_of::<Mat4>() + std::mem::size_of::<Vec4>() * 3;
		let stage2_pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![mat_tex_set_layout.clone()],
			push_constant_ranges: vec![PushConstantRange {
				stages: ShaderStages::VERTEX,
				offset: 0,
				size: stage2_push_constant_size.try_into().unwrap(),
			}],
			..Default::default()
		};
		let stage2_pipeline_layout = PipelineLayout::new(device.clone(), stage2_pipeline_layout_info)?;

		let moments_formats = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![
				Some(Format::R32G32B32A32_SFLOAT),
				Some(Format::R32_SFLOAT),
				//Some(Format::R32_SFLOAT),
			],
			depth_attachment_format: Some(depth_stencil_format),
			..Default::default()
		};
		let moments_pl = super::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleList,
			&[vs_nonorm::load(device.clone())?, fs_moments::load(device.clone())?],
			RasterizationState {
				cull_mode: CullMode::Back,
				..Default::default()
			},
			stage2_pipeline_layout,
			moments_formats,
			Some(moments_color_blend_state),
			Some(moments_depth_stencil_state),
		)?;

		//
		/* Stage 3: Calculate weights */
		//
		// The pipeline for stage 3 depends on the material of each mesh, so they're created outside
		// of this transparency renderer. They'll take the following descriptor set, which contains
		// the images rendered in Stage 2.
		let input_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
		};
		let stage3_inputs_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, input_binding.clone()), // moments
				(1, input_binding.clone()), // optical_depth
			]
			.into(),
			..Default::default()
		};
		let stage3_inputs_layout = DescriptorSetLayout::new(device.clone(), stage3_inputs_layout_info)?;

		//
		/* Stage 4: Composite transparency image onto opaque image */
		//
		let stage4_inputs_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, input_binding.clone()), // accum
				(1, input_binding),         // revealage
			]
			.into(),
			..Default::default()
		};
		let stage4_inputs_layout = DescriptorSetLayout::new(device.clone(), stage4_inputs_layout_info)?;
		let stage4_pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![stage4_inputs_layout.clone()],
			..Default::default()
		};
		let stage4_pipeline_layout = PipelineLayout::new(device.clone(), stage4_pipeline_layout_info)?;

		let stage4_blend = ColorBlendAttachmentState {
			blend: Some(AttachmentBlend::alpha()),
			..Default::default()
		};
		let stage4_color_blend_state = ColorBlendState::with_attachment_states(1, stage4_blend);

		let stencil_op_state = StencilOpState {
			ops: StencilOps {
				compare_op: CompareOp::Less,
				..Default::default()
			},
			reference: 0,
			..Default::default()
		};
		let stage4_depth_stencil_state = DepthStencilState {
			stencil: Some(StencilState {
				front: stencil_op_state,
				back: stencil_op_state,
			}),
			..Default::default()
		};

		let stage4_rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
			stencil_attachment_format: Some(depth_stencil_format),
			..Default::default()
		};
		let transparency_compositing_pl = super::pipeline::new(
			device.clone(),
			PrimitiveTopology::TriangleList,
			&[
				vs_fill_viewport::load(device.clone())?,
				fs_oit_compositing::load(device.clone())?,
			],
			RasterizationState::default(),
			stage4_pipeline_layout,
			stage4_rendering_info,
			Some(stage4_color_blend_state),
			Some(stage4_depth_stencil_state),
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
	) -> crate::Result<()>
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
	) -> crate::Result<()>
	{
		let moments_cb;
		if let Some(m_cb) = self.transparency_moments_cb.lock().unwrap().take() {
			moments_cb = m_cb;
		} else {
			// Skip OIT processing if no transparent materials are in view
			return Ok(());
		}
		let transparency_cb = self.transparency_cb.lock().unwrap().take().unwrap();

		let stage2_rendering_info = RenderingInfo {
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
				/*Some(RenderingAttachmentInfo {
					// min_depth
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.min_depth.clone())
				}),*/
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
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(depth_image.clone())
			}),
			stencil_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store,
				clear_value: Some(ClearValue::Stencil(0)),
				..RenderingAttachmentInfo::image_view(depth_image.clone())
			}),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		};

		let stage4_rendering_info = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(color_image.clone())
			})],
			stencil_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::DontCare,
				..RenderingAttachmentInfo::image_view(depth_image.clone())
			}),
			contents: SubpassContents::Inline,
			..Default::default()
		};

		let img_extent = self.images.moments.image().extent();
		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [img_extent[0] as f32, img_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};

		// draw the objects to the transparency framebuffer, then composite the transparency onto the main framebuffer
		let compositing_layout = self.transparency_compositing_pl.layout().clone();
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
				compositing_layout,
				0,
				vec![self.stage4_inputs.clone()],
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
	moments: Arc<ImageView>,
	optical_depth: Arc<ImageView>,
	//min_depth: Arc<ImageView>,
	accum: Arc<ImageView>,
	revealage: Arc<ImageView>,
}

fn create_mboit_images(
	memory_allocator: Arc<StandardMemoryAllocator>,
	descriptor_set_allocator: &StandardDescriptorSetAllocator,
	extent: [u32; 2],
	stage3_inputs_layout: Arc<DescriptorSetLayout>,
	stage4_inputs_layout: Arc<DescriptorSetLayout>,
) -> crate::Result<(MomentImageBundle, Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>)>
{
	let image_formats = [
		Format::R32G32B32A32_SFLOAT, // moments
		Format::R32_SFLOAT,          // optical_depth
		//Format::R32_SFLOAT,          // min_depth
		Format::R16G16B16A16_SFLOAT, // accum
		Format::R8_UNORM,            // revealage
	];

	let mut views = Vec::with_capacity(5);
	for format in image_formats {
		let info = ImageCreateInfo {
			usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
			format,
			extent: [extent[0], extent[1], 1],
			..Default::default()
		};
		let new_image = Image::new(memory_allocator.clone(), info.clone(), AllocationCreateInfo::default())?;
		views.push(ImageView::new_default(new_image)?);
	}

	let image_bundle = MomentImageBundle {
		moments: views[0].clone(),
		optical_depth: views[1].clone(),
		//min_depth: views[2].clone(),
		accum: views[2].clone(),
		revealage: views[3].clone(),
	};

	let stage3_inputs = PersistentDescriptorSet::new(
		descriptor_set_allocator,
		stage3_inputs_layout,
		[
			WriteDescriptorSet::image_view(0, views[0].clone()),
			WriteDescriptorSet::image_view(1, views[1].clone()),
			//WriteDescriptorSet::image_view(2, views[2].clone()),
		],
		[],
	)?;

	let stage4_inputs = PersistentDescriptorSet::new(
		descriptor_set_allocator,
		stage4_inputs_layout,
		[
			WriteDescriptorSet::image_view(0, views[2].clone()),
			WriteDescriptorSet::image_view(1, views[3].clone()),
		],
		[],
	)?;

	Ok((image_bundle, stage3_inputs, stage4_inputs))
}
