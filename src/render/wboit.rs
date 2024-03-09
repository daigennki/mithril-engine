/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use glam::*;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use vulkano::command_buffer::*;
use vulkano::descriptor_set::{allocator::*, layout::*, *};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{view::ImageView, *};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::graphics::{
	color_blend::*, depth_stencil::*, multisample::MultisampleState, subpass::PipelineRenderingCreateInfo, viewport::Viewport,
	*,
};
use vulkano::pipeline::{layout::*, *};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

mod vs_compositing
{
	vulkano_shaders::shader! {
		ty: "vertex",
		path: "src/shaders/fill_viewport.vert.glsl",
	}
}
mod fs_compositing
{
	vulkano_shaders::shader! {
		ty: "fragment",
		path: "src/shaders/wboit_compositing.frag.glsl"
	}
}
mod fs_compositing_ms
{
	vulkano_shaders::shader! {
		ty: "fragment",
		path: "src/shaders/wboit_compositing.frag.glsl",
		define: [("MULTISAMPLE", "")]
	}
}

/// A renderer that implements Order-Independent Transparency (OIT) using the Weighted Blended
/// Order-Independent Transparency (McGuire and Bavoil, 2013) algorithm.
///
/// Algorithm details: https://jcgt.org/published/0002/02/09/
pub struct WboitRenderer
{
	accum_image: Arc<ImageView>,
	revealage_image: Arc<ImageView>,
	compositing_pipeline: Arc<GraphicsPipeline>,
	descriptor_set_allocator: StandardDescriptorSetAllocator,
	weights_images: Arc<PersistentDescriptorSet>,
	transparency_cb: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,
}
impl WboitRenderer
{
	pub fn new(
		memory_allocator: Arc<StandardMemoryAllocator>,
		dimensions: [u32; 2],
		rasterization_samples: SampleCount,
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
		/* Composite transparency image onto opaque image */
		//
		// (weights are calculated by a shader that depends on each material, so we don't do any
		// setup related to that here)
		let input_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
		};
		let weights_images_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: BTreeMap::from([
				(0, input_binding.clone()), // accum
				(1, input_binding),         // revealage
			]),
			..Default::default()
		};
		let weights_images_layout = DescriptorSetLayout::new(device.clone(), weights_images_layout_info)?;
		let compositing_pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![weights_images_layout.clone()],
			..Default::default()
		};
		let compositing_pipeline_layout = PipelineLayout::new(device.clone(), compositing_pipeline_layout_info)?;

		let compositing_blend = ColorBlendAttachmentState {
			blend: Some(AttachmentBlend {
				src_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
				dst_color_blend_factor: BlendFactor::SrcAlpha,
				src_alpha_blend_factor: BlendFactor::Zero,
				dst_alpha_blend_factor: BlendFactor::One,
				..Default::default()
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

		let compositing_fs = if rasterization_samples != SampleCount::Sample1 {
			fs_compositing_ms::load(device.clone())?
		} else {
			fs_compositing::load(device.clone())?
		};

		let compositing_rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
			stencil_attachment_format: Some(depth_stencil_format),
			..Default::default()
		};
		let compositing_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				PipelineShaderStageCreateInfo::new(vs_compositing::load(device.clone())?.entry_point("main").unwrap()),
				PipelineShaderStageCreateInfo::new(compositing_fs.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(Default::default()),
			input_assembly_state: Some(Default::default()),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(Default::default()),
			multisample_state: Some(MultisampleState {
				rasterization_samples,
				..Default::default()
			}),
			depth_stencil_state: Some(compositing_depth_stencil_state),
			color_blend_state: Some(compositing_color_blend_state),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(compositing_rendering_info.into()),
			..GraphicsPipelineCreateInfo::layout(compositing_pipeline_layout)
		};
		let compositing_pipeline = GraphicsPipeline::new(device, None, compositing_pipeline_info)?;

		/* Create the images and descriptor sets */
		let (accum_image, revealage_image, weights_images) = create_images(
			memory_allocator,
			&descriptor_set_allocator,
			dimensions,
			rasterization_samples,
			weights_images_layout,
		)?;

		Ok(Self {
			accum_image,
			revealage_image,
			compositing_pipeline,
			descriptor_set_allocator,
			weights_images,
			transparency_cb: Mutex::new(None),
		})
	}

	/// Resize the output image to match a resized color image.
	fn resize_image(&mut self, memory_allocator: Arc<StandardMemoryAllocator>, dimensions: [u32; 2]) -> crate::Result<()>
	{
		let (accum, revealage, weights_images_set) = create_images(
			memory_allocator,
			&self.descriptor_set_allocator,
			dimensions,
			self.accum_image.image().samples(),
			self.weights_images.layout().clone(),
		)?;

		self.accum_image = accum;
		self.revealage_image = revealage;
		self.weights_images = weights_images_set;
		Ok(())
	}

	/// Do the OIT processing using the secondary command buffer that has already been received.
	pub fn process_transparency(
		&mut self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		color_image: Arc<ImageView>,
		depth_stencil_image: Arc<ImageView>,
		memory_allocator: Arc<StandardMemoryAllocator>,
	) -> crate::Result<()>
	{
		let img_extent = color_image.image().extent();
		if self.accum_image.image().extent() != img_extent {
			self.resize_image(memory_allocator, [img_extent[0], img_extent[1]])?;
		}

		let weights_cb = match self.transparency_cb.lock().unwrap().take() {
			Some(cb) => cb,
			None => return Ok(()), // Skip OIT processing if no transparent submeshes are in view
		};

		let weights_rendering_info = RenderingInfo {
			color_attachments: vec![
				Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.accum_image.clone())
				}),
				Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.revealage_image.clone())
				}),
			],
			depth_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(depth_stencil_image.clone())
			}),
			stencil_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store,
				clear_value: Some(ClearValue::Stencil(0)),
				..RenderingAttachmentInfo::image_view(depth_stencil_image.clone())
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
				..RenderingAttachmentInfo::image_view(depth_stencil_image)
			}),
			..Default::default()
		};

		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [img_extent[0] as f32, img_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};

		let compositing_layout = self.compositing_pipeline.layout().clone();
		let compositing_sets = vec![self.weights_images.clone()];
		cb.begin_rendering(weights_rendering_info)?
			.execute_commands(weights_cb)?
			.end_rendering()?
			.begin_rendering(compositing_rendering_info)?
			.set_viewport(0, [viewport].as_slice().into())?
			.bind_pipeline_graphics(self.compositing_pipeline.clone())?
			.bind_descriptor_sets(PipelineBindPoint::Graphics, compositing_layout, 0, compositing_sets)?
			.draw(3, 1, 0, 0)?
			.end_rendering()?;

		Ok(())
	}

	pub fn add_transparency_cb(&self, weights: Arc<SecondaryAutoCommandBuffer>)
	{
		*self.transparency_cb.lock().unwrap() = Some(weights)
	}
}

fn create_images(
	memory_allocator: Arc<StandardMemoryAllocator>,
	descriptor_set_allocator: &StandardDescriptorSetAllocator,
	extent: [u32; 2],
	rasterization_samples: SampleCount,
	weights_images_layout: Arc<DescriptorSetLayout>,
) -> crate::Result<(Arc<ImageView>, Arc<ImageView>, Arc<PersistentDescriptorSet>)>
{
	let accum_info = ImageCreateInfo {
		usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
		format: Format::R16G16B16A16_SFLOAT,
		extent: [extent[0], extent[1], 1],
		samples: rasterization_samples,
		..Default::default()
	};
	let accum_image = Image::new(memory_allocator.clone(), accum_info.clone(), AllocationCreateInfo::default())?;
	let accum = ImageView::new_default(accum_image)?;

	let revealage_info = ImageCreateInfo {
		format: Format::R8_UNORM,
		..accum_info
	};
	let revealage_image = Image::new(memory_allocator.clone(), revealage_info, AllocationCreateInfo::default())?;
	let revealage = ImageView::new_default(revealage_image)?;

	let weights_images = PersistentDescriptorSet::new(
		descriptor_set_allocator,
		weights_images_layout,
		[
			WriteDescriptorSet::image_view(0, accum.clone()),
			WriteDescriptorSet::image_view(1, revealage.clone()),
		],
		[],
	)?;

	Ok((accum, revealage, weights_images))
}
