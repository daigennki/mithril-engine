/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, BlitImageInfo, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{
	allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::Device;
use vulkano::format::{Format, FormatFeatures};
use vulkano::image::{view::ImageView, Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::{
	compute::ComputePipelineCreateInfo, layout::PipelineLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint,
	PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::ShaderStages;
use vulkano::swapchain::ColorSpace;

// Some notes regarding observed support (with AMD, Intel, and NVIDIA GPUs) for depth/stencil formats:
//
// - `D16_UNORM`: Supported on all GPUs.
// - `D16_UNORM_S8_UINT`: Only supported on AMD GPUs.
// - `X8_D24_UNORM_PACK32`: Only supported on NVIDIA and Intel GPUs.
// - `D24_UNORM_S8_UINT`: Only supported on NVIDIA and Intel GPUs.
// - `D32_SFLOAT`: Supported on all GPUs.
// - `D32_SFLOAT_S8_UINT`: Supported on all GPUs.
// - `S8_UINT`: Only supported on AMD GPUs. Possibly supported on NVIDIA GPUs.
//
// (source: https://vulkan.gpuinfo.org/listoptimaltilingformats.php)
const DEPTH_STENCIL_FORMAT_CANDIDATES: [Format; 2] = [Format::D24_UNORM_S8_UINT, Format::D16_UNORM_S8_UINT];

mod compute_gamma
{
	vulkano_shaders::shader! {
		ty: "compute",
		src: r"
			#version 460

			layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

			layout(binding = 0, rgba16f) uniform restrict image2D color_image;

			void main()
			{
				const float gamma = 1.0 / 2.2;

				ivec2 image_coord = ivec2(gl_GlobalInvocationID.xy);
				vec3 rgb_lin = imageLoad(color_image, image_coord).rgb;

				float r = pow(rgb_lin.r, gamma);
				float g = pow(rgb_lin.g, gamma);
				float b = pow(rgb_lin.b, gamma);

				imageStore(color_image, image_coord, vec4(r, g, b, 1.0));
			}
		",
	}
}

/// Things related to the output color/depth images and gamma correction.
pub struct RenderTarget
{
	color_set_allocator: StandardDescriptorSetAllocator,
	set_layout: Arc<DescriptorSetLayout>,
	depth_stencil_format: Format,
	color_image: Option<Arc<ImageView>>,
	depth_image: Option<Arc<ImageView>>,
	color_set: Option<Arc<PersistentDescriptorSet>>, // Contains `color_image` as a storage image binding.
	gamma_pipeline: Arc<ComputePipeline>,            // The gamma correction pipeline.
}
impl RenderTarget
{
	pub fn new(device: Arc<Device>) -> crate::Result<Self>
	{
		let set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 2,
			..Default::default()
		};
		let color_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), set_alloc_info);

		let depth_stencil_format = DEPTH_STENCIL_FORMAT_CANDIDATES
			.into_iter()
			.find(|format| {
				device
					.physical_device()
					.format_properties(*format)
					.unwrap()
					.optimal_tiling_features
					.contains(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
			})
			.ok_or("none of the depth/stencil format candidates are supported")?;
		log::debug!("using depth/stencil format {depth_stencil_format:?}");

		let image_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::COMPUTE,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
		};
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [(0, image_binding)].into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

		let pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![set_layout.clone()],
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_info)?;
		let entry_point = compute_gamma::load(device.clone())?.entry_point("main").unwrap();
		let stage = PipelineShaderStageCreateInfo::new(entry_point);
		let pipeline_create_info = ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout);
		let gamma_pipeline = ComputePipeline::new(device.clone(), None, pipeline_create_info)?;

		Ok(Self {
			color_set_allocator,
			set_layout,
			depth_stencil_format,
			color_image: None,
			depth_image: None,
			color_set: None,
			gamma_pipeline,
		})
	}

	/// Get the color and depth images. They'll be resized before being returned if `extent` changed.
	pub fn get_images(
		&mut self,
		memory_allocator: Arc<StandardMemoryAllocator>,
		extent2: [u32; 2],
	) -> crate::Result<(Arc<ImageView>, Arc<ImageView>)>
	{
		let extent = [extent2[0], extent2[1], 1];
		if Some(extent) != self.color_image.as_ref().map(|view| view.image().extent()) {
			let color_create_info = ImageCreateInfo {
				format: Format::R16G16B16A16_SFLOAT,
				extent,
				usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
				..Default::default()
			};
			let color_image = Image::new(memory_allocator.clone(), color_create_info, AllocationCreateInfo::default())?;
			let color = ImageView::new_default(color_image)?;
			self.color_image = Some(color.clone());

			let depth_create_info = ImageCreateInfo {
				format: self.depth_stencil_format,
				extent,
				usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
				..Default::default()
			};
			let depth_image = Image::new(memory_allocator, depth_create_info, AllocationCreateInfo::default())?;
			let depth = ImageView::new_default(depth_image)?;
			self.depth_image = Some(depth);

			let set_layout = self.set_layout.clone();
			let set_write = [WriteDescriptorSet::image_view(0, color)];
			let set = PersistentDescriptorSet::new(&self.color_set_allocator, set_layout, set_write, [])?;
			self.color_set = Some(set);
		}

		Ok((self.color_image.clone().unwrap(), self.depth_image.clone().unwrap()))
	}

	pub fn depth_stencil_format(&self) -> Format
	{
		self.depth_stencil_format
	}

	/// Blit the color image to the swapchain image, after converting it to the swapchain color
	/// space if necessary.
	pub fn blit_to_swapchain(
		&self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		swapchain_image: Arc<Image>,
		color_space: ColorSpace,
	) -> crate::Result<()>
	{
		let color_image = self.color_image.as_ref().unwrap().image().clone();

		if color_space == ColorSpace::SrgbNonLinear {
			// perform gamma correction
			let image_extent = color_image.extent();
			let workgroups_x = image_extent[0].div_ceil(64);
			let layout = self.gamma_pipeline.layout().clone();
			cb.bind_pipeline_compute(self.gamma_pipeline.clone())?
				.bind_descriptor_sets(PipelineBindPoint::Compute, layout, 0, self.color_set.clone().unwrap())?
				.dispatch([workgroups_x, image_extent[1], 1])?;
		}

		cb.blit_image(BlitImageInfo::images(color_image, swapchain_image))?;

		Ok(())
	}
}
