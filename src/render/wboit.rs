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
use vulkano::device::{Device, DeviceOwned};
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
		define: [("MULTISAMPLED_IMAGE", "")]
	}
}

/// A renderer that implements Order-Independent Transparency (OIT) using the Weighted Blended
/// Order-Independent Transparency (McGuire and Bavoil, 2013) algorithm.
///
/// Algorithm details: https://jcgt.org/published/0002/02/09/
pub struct WboitRenderer
{
	compositing_pipeline_layout: Arc<PipelineLayout>,
	compositing_pipeline: Option<Arc<GraphicsPipeline>>,
	descriptor_set_allocator: StandardDescriptorSetAllocator,

	accum_image: Option<Arc<ImageView>>,
	revealage_image: Option<Arc<ImageView>>,
	weight_set: Option<Arc<PersistentDescriptorSet>>,

	transparency_cb: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,
}
impl WboitRenderer
{
	pub fn new(device: Arc<Device>) -> crate::Result<Self>
	{
		let set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 2,
			..Default::default()
		};
		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), set_alloc_info);

		// Compositing pipeline setup (weights are calculated by a shader that depends on each
		// material, so we don't do any setup related to that here)
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

		Ok(Self {
			compositing_pipeline_layout,
			compositing_pipeline: None,
			descriptor_set_allocator,
			accum_image: None,
			revealage_image: None,
			weight_set: None,
			transparency_cb: Mutex::new(None),
		})
	}

	/// Resize the output image to match a resized color image.
	fn resize_image(
		&mut self,
		memory_allocator: Arc<StandardMemoryAllocator>,
		extent: [u32; 3],
		rasterization_samples: SampleCount,
		depth_stencil_format: Format,
	) -> crate::Result<()>
	{
		// Also re-create the pipeline if the multisample configuration has changed.
		let prev_samples = self
			.compositing_pipeline
			.as_ref()
			.and_then(|pl| pl.multisample_state().map(|state| state.rasterization_samples));
		if prev_samples != Some(rasterization_samples) {
			self.compositing_pipeline = Some(create_compositing_pipeline(
				self.compositing_pipeline_layout.clone(),
				rasterization_samples,
				depth_stencil_format,
			)?);
		}

		let accum_info = ImageCreateInfo {
			usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
			format: Format::R16G16B16A16_SFLOAT,
			extent,
			samples: rasterization_samples,
			..Default::default()
		};
		let accum_image = Image::new(memory_allocator.clone(), accum_info.clone(), AllocationCreateInfo::default())?;
		let accum_view = ImageView::new_default(accum_image)?;
		self.accum_image = Some(accum_view.clone());

		let revealage_info = ImageCreateInfo {
			format: Format::R8_UNORM,
			..accum_info
		};
		let revealage_image = Image::new(memory_allocator.clone(), revealage_info, AllocationCreateInfo::default())?;
		let revealage_view = ImageView::new_default(revealage_image)?;
		self.revealage_image = Some(revealage_view.clone());

		let writes = [
			WriteDescriptorSet::image_view(0, accum_view),
			WriteDescriptorSet::image_view(1, revealage_view),
		];
		let set_layout = self.compositing_pipeline_layout.set_layouts()[0].clone();
		self.weight_set = Some(PersistentDescriptorSet::new(
			&self.descriptor_set_allocator,
			set_layout,
			writes,
			[],
		)?);

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
		let weights_cb = match self.transparency_cb.lock().unwrap().take() {
			Some(cb) => cb,
			None => return Ok(()), // Skip OIT processing if no transparent submeshes are in view
		};

		let img_extent = color_image.image().extent();
		let samples = color_image.image().samples();
		let (prev_extent, prev_samples) = self
			.accum_image
			.as_ref()
			.map(|view| (view.image().extent(), view.image().samples()))
			.unzip();
		if prev_extent != Some(img_extent) || prev_samples != Some(samples) {
			self.resize_image(memory_allocator, img_extent, samples, depth_stencil_image.format())?;
		}

		let weights_rendering_info = RenderingInfo {
			color_attachments: vec![
				Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.accum_image.clone().unwrap())
				}),
				Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(self.revealage_image.clone().unwrap())
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

		let compositing_layout = self.compositing_pipeline_layout.clone();
		let compositing_set = self.weight_set.clone().unwrap();
		cb.begin_rendering(weights_rendering_info)?
			.execute_commands(weights_cb)?
			.end_rendering()?
			.begin_rendering(compositing_rendering_info)?
			.set_viewport(0, smallvec::smallvec![viewport])?
			.bind_pipeline_graphics(self.compositing_pipeline.clone().unwrap())?
			.bind_descriptor_sets(PipelineBindPoint::Graphics, compositing_layout, 0, compositing_set)?
			.draw(3, 1, 0, 0)?
			.end_rendering()?;

		Ok(())
	}

	pub fn add_transparency_cb(&self, weights: Arc<SecondaryAutoCommandBuffer>)
	{
		*self.transparency_cb.lock().unwrap() = Some(weights)
	}
}

fn create_compositing_pipeline(
	pipeline_layout: Arc<PipelineLayout>,
	rasterization_samples: SampleCount,
	depth_stencil_format: Format,
) -> crate::Result<Arc<GraphicsPipeline>>
{
	let device = pipeline_layout.device().clone();
	let blend = ColorBlendAttachmentState {
		blend: Some(AttachmentBlend {
			src_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
			dst_color_blend_factor: BlendFactor::SrcAlpha,
			src_alpha_blend_factor: BlendFactor::Zero,
			dst_alpha_blend_factor: BlendFactor::One,
			..Default::default()
		}),
		..Default::default()
	};
	let color_blend_state = ColorBlendState::with_attachment_states(1, blend);

	let stencil_op_state = StencilOpState {
		ops: StencilOps {
			compare_op: CompareOp::Less,
			..Default::default()
		},
		reference: 0,
		..Default::default()
	};
	let depth_stencil_state = DepthStencilState {
		stencil: Some(StencilState {
			front: stencil_op_state,
			back: stencil_op_state,
		}),
		..Default::default()
	};

	let fs = if rasterization_samples != SampleCount::Sample1 {
		fs_compositing_ms::load(device.clone())?
	} else {
		fs_compositing::load(device.clone())?
	};

	let rendering_info = PipelineRenderingCreateInfo {
		color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
		stencil_attachment_format: Some(depth_stencil_format),
		..Default::default()
	};
	let pipeline_info = GraphicsPipelineCreateInfo {
		stages: smallvec::smallvec![
			PipelineShaderStageCreateInfo::new(vs_compositing::load(device.clone())?.entry_point("main").unwrap()),
			PipelineShaderStageCreateInfo::new(fs.entry_point("main").unwrap()),
		],
		vertex_input_state: Some(Default::default()),
		input_assembly_state: Some(Default::default()),
		viewport_state: Some(Default::default()),
		rasterization_state: Some(Default::default()),
		multisample_state: Some(MultisampleState {
			rasterization_samples,
			..Default::default()
		}),
		depth_stencil_state: Some(depth_stencil_state),
		color_blend_state: Some(color_blend_state),
		dynamic_state: [DynamicState::Viewport].into_iter().collect(),
		subpass: Some(rendering_info.into()),
		..GraphicsPipelineCreateInfo::layout(pipeline_layout)
	};

	Ok(GraphicsPipeline::new(device, None, pipeline_info)?)
}
