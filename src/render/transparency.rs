/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo, SecondaryAutoCommandBuffer,
	SubpassContents,
};
use vulkano::descriptor_set::{
	allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{view::ImageView, Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::graphics::{
	color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState},
	depth_stencil::{CompareOp, DepthState, DepthStencilState, StencilOpState, StencilOps, StencilState},
	rasterization::{CullMode, RasterizationState},
	subpass::PipelineRenderingCreateInfo,
	vertex_input::VertexInputState,
	viewport::Viewport,
	GraphicsPipeline, GraphicsPipelineCreateInfo,
};
use vulkano::pipeline::{
	layout::{PipelineLayoutCreateInfo, PushConstantRange},
	DynamicState, PipelineLayout, PipelineShaderStageCreateInfo,
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

	descriptor_set_allocator: StandardDescriptorSetAllocator,
	moments_images: Arc<PersistentDescriptorSet>,
	weights_images: Arc<PersistentDescriptorSet>,

	transparency_cb: Mutex<Option<(Arc<SecondaryAutoCommandBuffer>, Arc<SecondaryAutoCommandBuffer>)>>,
}
impl MomentTransparencyRenderer
{
	pub fn new(
		memory_allocator: Arc<StandardMemoryAllocator>,
		mat_tex_set_layout: Arc<DescriptorSetLayout>,
		dimensions: [u32; 2],
		depth_stencil_format: Format,
	) -> crate::Result<Self>
	{
		let device = memory_allocator.device().clone();
		let set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 2,
			..Default::default()
		};
		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), set_alloc_info);

		//
		/* Stage 2: Calculate moments */
		//
		let all_additive_blend = ColorBlendAttachmentState {
			blend: Some(AttachmentBlend {
				alpha_blend_op: BlendOp::Add,
				..AttachmentBlend::additive()
			}),
			..Default::default()
		};
		let moments_color_blend_state = ColorBlendState::with_attachment_states(2, all_additive_blend);

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
			color_attachment_formats: vec![Some(Format::R32G32B32A32_SFLOAT), Some(Format::R32_SFLOAT)],
			depth_attachment_format: Some(depth_stencil_format),
			..Default::default()
		};
		let vertex_input_state = VertexInputState {
			bindings: (0..).zip(super::model::VERTEX_BINDINGS).collect(),
			attributes: (0..).zip(super::model::VERTEX_ATTRIBUTES).collect(),
			..Default::default()
		};
		let stage2_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				PipelineShaderStageCreateInfo::new(vs_nonorm::load(device.clone())?.entry_point("main").unwrap()),
				PipelineShaderStageCreateInfo::new(fs_moments::load(device.clone())?.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(vertex_input_state),
			input_assembly_state: Some(Default::default()),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(RasterizationState {
				cull_mode: CullMode::Back,
				..Default::default()
			}),
			multisample_state: Some(Default::default()),
			depth_stencil_state: Some(moments_depth_stencil_state),
			color_blend_state: Some(moments_color_blend_state),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(moments_formats.into()),
			..GraphicsPipelineCreateInfo::layout(stage2_pipeline_layout)
		};
		let moments_pl = GraphicsPipeline::new(device.clone(), None, stage2_pipeline_info)?;

		//
		/* Stage 3: Calculate weights */
		//
		// The pipeline for stage 3 depends on the material of each mesh, so they're created outside
		// of this transparency renderer. They'll take a descriptor set with the following layout,
		// which contains the images rendered in Stage 2.
		let input_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
		};
		let moments_images_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: BTreeMap::from([
				(0, input_binding.clone()), // moments
				(1, input_binding.clone()), // optical_depth
			]),
			..Default::default()
		};
		let moments_images_layout = DescriptorSetLayout::new(device.clone(), moments_images_layout_info)?;

		//
		/* Stage 4: Composite transparency image onto opaque image */
		//
		let weights_images_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: BTreeMap::from([
				(0, input_binding.clone()), // accum
				(1, input_binding),         // revealage
			]),
			..Default::default()
		};
		let weights_images_layout = DescriptorSetLayout::new(device.clone(), weights_images_layout_info)?;
		let stage4_pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![weights_images_layout.clone()],
			..Default::default()
		};
		let stage4_pipeline_layout = PipelineLayout::new(device.clone(), stage4_pipeline_layout_info)?;

		let compositing_blend = ColorBlendAttachmentState {
			blend: Some(AttachmentBlend {
				src_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
				dst_color_blend_factor: BlendFactor::SrcAlpha,
				color_blend_op: BlendOp::Add,
				..AttachmentBlend::ignore_source()
			}),
			..Default::default()
		};
		let compositing_color_blend_state = ColorBlendState::with_attachment_states(1, compositing_blend);

		let stencil_op_state = StencilOpState {
			ops: StencilOps {
				compare_op: CompareOp::Less,
				..Default::default()
			},
			reference: 0,
			..Default::default()
		};
		let compositing_depth_stencil_state = DepthStencilState {
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
		let stage4_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				PipelineShaderStageCreateInfo::new(vs_fill_viewport::load(device.clone())?.entry_point("main").unwrap()),
				PipelineShaderStageCreateInfo::new(fs_oit_compositing::load(device.clone())?.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(Default::default()),
			input_assembly_state: Some(Default::default()),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(Default::default()),
			multisample_state: Some(Default::default()),
			depth_stencil_state: Some(compositing_depth_stencil_state),
			color_blend_state: Some(compositing_color_blend_state),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(stage4_rendering_info.into()),
			..GraphicsPipelineCreateInfo::layout(stage4_pipeline_layout)
		};
		let transparency_compositing_pl = GraphicsPipeline::new(device, None, stage4_pipeline_info)?;

		/* Create the images and descriptor sets */
		let (images, moments_images, weights_images) = create_mboit_images(
			memory_allocator,
			&descriptor_set_allocator,
			dimensions,
			moments_images_layout,
			weights_images_layout,
		)?;

		Ok(MomentTransparencyRenderer {
			images,
			moments_pl,
			transparency_compositing_pl,
			descriptor_set_allocator,
			moments_images,
			weights_images,
			transparency_cb: Mutex::new(None),
		})
	}

	/// Resize the output image to match a resized color image.
	fn resize_image(&mut self, memory_allocator: Arc<StandardMemoryAllocator>, dimensions: [u32; 2]) -> crate::Result<()>
	{
		let (images, moments_images, weights_images) = create_mboit_images(
			memory_allocator,
			&self.descriptor_set_allocator,
			dimensions,
			self.moments_images.layout().clone(),
			self.weights_images.layout().clone(),
		)?;

		self.images = images;
		self.moments_images = moments_images;
		self.weights_images = weights_images;
		Ok(())
	}

	/// Do the OIT processing using the secondary command buffers that have already been received.
	pub fn process_transparency(
		&mut self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		color_image: Arc<ImageView>,
		depth_image: Arc<ImageView>,
		memory_allocator: Arc<StandardMemoryAllocator>,
	) -> crate::Result<()>
	{
		let img_extent = color_image.image().extent();
		if self.images.accum.image().extent() != img_extent {
			self.resize_image(memory_allocator, [img_extent[0], img_extent[1]])?;
		}

		let (moments_cb, weights_cb) = match self.transparency_cb.lock().unwrap().take() {
			Some(cb) => cb,
			None => return Ok(()), // Skip OIT processing if no transparent submeshes are in view
		};

		let depth_attachment = Some(RenderingAttachmentInfo {
			load_op: AttachmentLoadOp::Load,
			store_op: AttachmentStoreOp::Store,
			..RenderingAttachmentInfo::image_view(depth_image.clone())
		});

		let moments_rendering_info = RenderingInfo {
			color_attachments: vec![
				Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.moments.clone())
				}),
				Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.optical_depth.clone())
				}),
			],
			depth_attachment: depth_attachment.clone(),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		};

		let weights_rendering_info = RenderingInfo {
			color_attachments: vec![
				Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.accum.clone())
				}),
				Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.images.revealage.clone())
				}),
			],
			depth_attachment,
			stencil_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store,
				clear_value: Some(ClearValue::Stencil(0)),
				..RenderingAttachmentInfo::image_view(depth_image.clone())
			}),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		};

		let compositing_rendering_info = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(color_image.clone())
			})],
			stencil_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				..RenderingAttachmentInfo::image_view(depth_image)
			}),
			..Default::default()
		};

		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [img_extent[0] as f32, img_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};

		let compositing_layout = self.transparency_compositing_pl.layout().clone();
		let compositing_sets = vec![self.weights_images.clone()];
		cb.begin_rendering(moments_rendering_info)?
			.execute_commands(moments_cb)?
			.end_rendering()?
			.begin_rendering(weights_rendering_info)?
			.execute_commands(weights_cb)?
			.end_rendering()?
			.begin_rendering(compositing_rendering_info)?
			.set_viewport(0, [viewport].as_slice().into())?
			.bind_pipeline_graphics(self.transparency_compositing_pl.clone())?
			.bind_descriptor_sets(PipelineBindPoint::Graphics, compositing_layout, 0, compositing_sets)?
			.draw(3, 1, 0, 0)?
			.end_rendering()?;

		Ok(())
	}

	pub fn add_transparency_cb(&self, moments: Arc<SecondaryAutoCommandBuffer>, weights: Arc<SecondaryAutoCommandBuffer>)
	{
		*self.transparency_cb.lock().unwrap() = Some((moments, weights))
	}

	pub fn get_moments_pipeline(&self) -> &Arc<GraphicsPipeline>
	{
		&self.moments_pl
	}

	pub fn get_moments_images_set(&self) -> &Arc<PersistentDescriptorSet>
	{
		&self.moments_images
	}
}

struct MomentImageBundle
{
	moments: Arc<ImageView>,
	optical_depth: Arc<ImageView>,
	accum: Arc<ImageView>,
	revealage: Arc<ImageView>,
}

// if using min_depth, use this blending for it:
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
// and render to it with this:
/*Some(RenderingAttachmentInfo {
	load_op: AttachmentLoadOp::Clear,
	store_op: AttachmentStoreOp::Store,
	clear_value: Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])),
	..RenderingAttachmentInfo::image_view(self.images.min_depth.clone())
}),*/

fn create_mboit_images(
	memory_allocator: Arc<StandardMemoryAllocator>,
	descriptor_set_allocator: &StandardDescriptorSetAllocator,
	extent: [u32; 2],
	moments_images_layout: Arc<DescriptorSetLayout>,
	weights_images_layout: Arc<DescriptorSetLayout>,
) -> crate::Result<(MomentImageBundle, Arc<PersistentDescriptorSet>, Arc<PersistentDescriptorSet>)>
{
	let image_formats = [
		Format::R32G32B32A32_SFLOAT, // moments
		Format::R32_SFLOAT,          // optical_depth
		//Format::R32_SFLOAT,          // min_depth
		Format::R16G16B16A16_SFLOAT, // accum
		Format::R8_UNORM,            // revealage
	];

	let mut views = Vec::with_capacity(4);
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
		accum: views[2].clone(),
		revealage: views[3].clone(),
	};

	let moments_images = PersistentDescriptorSet::new(
		descriptor_set_allocator,
		moments_images_layout,
		[
			WriteDescriptorSet::image_view(0, views[0].clone()),
			WriteDescriptorSet::image_view(1, views[1].clone()),
		],
		[],
	)?;

	let weights_images = PersistentDescriptorSet::new(
		descriptor_set_allocator,
		weights_images_layout,
		[
			WriteDescriptorSet::image_view(0, views[2].clone()),
			WriteDescriptorSet::image_view(1, views[3].clone()),
		],
		[],
	)?;

	Ok((image_bundle, moments_images, weights_images))
}
