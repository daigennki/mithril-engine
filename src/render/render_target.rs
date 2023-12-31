/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, BlitImageInfo, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo, SubpassContents,
};
use vulkano::descriptor_set::{
	allocator::StandardDescriptorSetAllocator,
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{
	sampler::{Sampler, SamplerCreateInfo},
	view::ImageView,
	Image, ImageCreateInfo, ImageUsage,
};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::{
	graphics::{input_assembly::PrimitiveTopology, rasterization::RasterizationState, viewport::Viewport},
	layout::PipelineLayoutCreateInfo,
	GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;
use vulkano::swapchain::ColorSpace;

mod vs_fill_viewport
{
	vulkano_shaders::shader! {
		ty: "vertex",
		path: "src/shaders/fill_viewport.vert.glsl",
	}
}
mod fs_gamma
{
	vulkano_shaders::shader! {
		ty: "fragment",
		src: r"
			#version 450

			layout(binding = 0) uniform sampler2D color_in;

			layout(location = 0) in vec2 texcoord;
			layout(location = 0) out vec4 color_out;

			void main()
			{
				const float gamma = 1.0 / 2.2;
				vec3 rgb_lin = texture(color_in, texcoord).rgb;
				float r = pow(rgb_lin.r, gamma);
				float g = pow(rgb_lin.g, gamma);
				float b = pow(rgb_lin.b, gamma);
				color_out = vec4(r, g, b, 1.0);
			}
		",
	}
}

pub struct RenderTarget
{
	color_image: Arc<ImageView>, // An FP16, linear gamma image which everything will be rendered to.
	color_set: Arc<PersistentDescriptorSet>, // Descriptor set containing `color_image`.
	depth_image: Arc<ImageView>,

	// The pipeline used to apply gamma correction.
	// Only used when the output color space is `SrgbNonLinear`.
	gamma_pipeline: Option<Arc<GraphicsPipeline>>,
}
impl RenderTarget
{
	pub fn new(
		memory_allocator: Arc<StandardMemoryAllocator>,
		descriptor_set_allocator: &StandardDescriptorSetAllocator,
		dimensions: [u32; 2],
		swapchain_format: Format,
		swapchain_color_space: ColorSpace,
	) -> crate::Result<Self>
	{
		let device = memory_allocator.device().clone();

		let usage = (swapchain_color_space == ColorSpace::SrgbNonLinear)
			.then_some(ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED)
			.unwrap_or(ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC);

		let color_create_info = ImageCreateInfo {
			format: Format::R16G16B16A16_SFLOAT,
			extent: [dimensions[0], dimensions[1], 1],
			usage,
			..Default::default()
		};
		let color_image = Image::new(memory_allocator.clone(), color_create_info, AllocationCreateInfo::default())?;
		let color_image_view = ImageView::new_default(color_image)?;

		let depth_create_info = ImageCreateInfo {
			format: super::MAIN_DEPTH_FORMAT,
			extent: [dimensions[0], dimensions[1], 1],
			usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
			..Default::default()
		};
		let depth_image = Image::new(memory_allocator.clone(), depth_create_info, AllocationCreateInfo::default())?;
		let depth_image_view = ImageView::new_default(depth_image)?;

		let input_sampler = Sampler::new(device.clone(), SamplerCreateInfo::default())?;
		let input_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			immutable_samplers: vec![input_sampler],
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
		};
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [(0, input_binding)].into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;
		let color_set = PersistentDescriptorSet::new(
			descriptor_set_allocator,
			set_layout.clone(),
			[WriteDescriptorSet::image_view(0, color_image_view.clone())],
			[],
		)?;

		let pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![set_layout.clone()],
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_info)?;

		let gamma_pipeline = if swapchain_color_space == ColorSpace::SrgbNonLinear {
			Some(super::pipeline::new(
				device.clone(),
				PrimitiveTopology::TriangleList,
				&[
					vs_fill_viewport::load(device.clone()).unwrap(),
					fs_gamma::load(device).unwrap(),
				],
				RasterizationState::default(),
				pipeline_layout,
				&[(swapchain_format, None)],
				None,
				None,
			)?)
		} else {
			None
		};

		Ok(Self {
			color_image: color_image_view,
			color_set,
			depth_image: depth_image_view,
			gamma_pipeline,
		})
	}

	pub fn color_image(&self) -> &Arc<ImageView>
	{
		&self.color_image
	}
	pub fn depth_image(&self) -> &Arc<ImageView>
	{
		&self.depth_image
	}

	pub fn blit_to_swapchain(
		&self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		swapchain_image: Arc<ImageView>,
	) -> crate::Result<()>
	{
		if let Some(gamma_pipeline) = &self.gamma_pipeline {
			// perform gamma correction and write to the swapchain image
			let swapchain_image_extent = swapchain_image.image().extent();
			let render_info = RenderingInfo {
				color_attachments: vec![Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::DontCare,
					store_op: AttachmentStoreOp::Store,
					..RenderingAttachmentInfo::image_view(swapchain_image)
				})],
				contents: SubpassContents::Inline,
				..Default::default()
			};
			let viewport = Viewport {
				offset: [0.0, 0.0],
				extent: [swapchain_image_extent[0] as f32, swapchain_image_extent[1] as f32],
				depth_range: 0.0..=1.0,
			};

			cb.begin_rendering(render_info)?
				.set_viewport(0, [viewport].as_slice().into())?
				.bind_pipeline_graphics(gamma_pipeline.clone())?
				.bind_descriptor_sets(
					PipelineBindPoint::Graphics,
					gamma_pipeline.layout().clone(),
					0,
					vec![self.color_set.clone()],
				)?
				.draw(3, 1, 0, 0)?
				.end_rendering()?;
		} else {
			// blit to the swapchain image without gamma correction
			cb.blit_image(BlitImageInfo::images(
				self.color_image.image().clone(),
				swapchain_image.image().clone(),
			))?;
		}

		Ok(())
	}
}
