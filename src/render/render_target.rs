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
	DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{view::ImageView, Image, ImageCreateInfo, ImageFormatInfo, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::{
	compute::ComputePipelineCreateInfo, layout::PipelineLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint,
	PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::ShaderStages;
use vulkano::swapchain::ColorSpace;

// # Notes about format used for main depth/stencil buffer
//
// While [NVIDIA recommends](https://developer.nvidia.com/blog/vulkan-dos-donts/) using a
// 24-bit depth format (`D24_UNORM_S8_UINT`), it doesn't seem to be very well-supported outside of
// NVIDIA GPUs. Only about 70% of GPUs on Windows and 50% of GPUs on Linux seem to support it,
// while `D16_UNORM` and `D32_SFLOAT` both have 100% support.
//
// More notes regarding observed support for depth/stencil formats:
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

const DEPTH_STENCIL_FORMAT_CANDIDATES: [Format; 2] = [Format::D24_UNORM_S8_UINT, Format::D16_UNORM_S8_UINT];

pub struct RenderTarget
{
	descriptor_set_allocator: StandardDescriptorSetAllocator,

	color_image: Arc<ImageView>, // An FP16, linear gamma image which everything will be rendered to.
	depth_image: Arc<ImageView>,

	// The pipeline used to apply gamma correction, and the descriptor set that contains the image
	// it applies gamma correction to.
	// Only used when the output color space is `SrgbNonLinear`.
	gamma_pipeline: Arc<ComputePipeline>,
	color_set: Arc<PersistentDescriptorSet>,
}
impl RenderTarget
{
	pub fn new(memory_allocator: Arc<StandardMemoryAllocator>, extent: [u32; 2]) -> crate::Result<Self>
	{
		let device = memory_allocator.device().clone();
		let set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 2,
			..Default::default()
		};
		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), set_alloc_info);

		let mut selected_depth_format = None;
		for format in DEPTH_STENCIL_FORMAT_CANDIDATES {
			let image_format_info = ImageFormatInfo {
				format,
				usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
				..Default::default()
			};
			if device.physical_device().image_format_properties(image_format_info)?.is_some() {
				selected_depth_format = Some(format);
				break;
			}
		}
		let depth_format = selected_depth_format.ok_or("none of the depth/stencil format candidates are supported")?;
		log::debug!("using depth/stencil format {depth_format:?}");

		let extent3 = [extent[0], extent[1], 1];
		let (color_image_view, depth_image_view) = create_images(memory_allocator.clone(), extent3, depth_format)?;

		let image_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::COMPUTE,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
		};
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [(0, image_binding)].into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;
		let set_write = [WriteDescriptorSet::image_view(0, color_image_view.clone())];
		let color_set = PersistentDescriptorSet::new(&descriptor_set_allocator, set_layout.clone(), set_write, [])?;

		let pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![set_layout],
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_info)?;
		let entry_point = compute_gamma::load(device.clone())?.entry_point("main").unwrap();
		let stage = PipelineShaderStageCreateInfo::new(entry_point);
		let pipeline_create_info = ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout);
		let gamma_pipeline = ComputePipeline::new(device, None, pipeline_create_info)?;

		Ok(Self {
			descriptor_set_allocator,
			color_image: color_image_view,
			depth_image: depth_image_view,
			gamma_pipeline,
			color_set,
		})
	}

	/// Get the color and depth images. They will be resized before being returned if the given
	/// `extent` changed.
	pub fn get_images(
		&mut self,
		memory_allocator: Arc<StandardMemoryAllocator>,
		extent2: [u32; 2],
	) -> crate::Result<(Arc<ImageView>, Arc<ImageView>)>
	{
		let extent = [extent2[0], extent2[1], 1];
		if extent != self.color_image.image().extent() {
			let (color, depth) = create_images(memory_allocator.clone(), extent, self.depth_image.format())?;
			self.color_image = color;
			self.depth_image = depth;

			let set_layout = self.color_set.layout().clone();
			let set_write = [WriteDescriptorSet::image_view(0, self.color_image.clone())];
			self.color_set = PersistentDescriptorSet::new(&self.descriptor_set_allocator, set_layout, set_write, [])?;
		}

		Ok((self.color_image.clone(), self.depth_image.clone()))
	}

	pub fn depth_stencil_format(&self) -> Format
	{
		self.depth_image.format()
	}

	pub fn blit_to_swapchain(
		&self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		swapchain_image: Arc<Image>,
		color_space: ColorSpace,
	) -> crate::Result<()>
	{
		// for rendering to non-linear sRGB, perform gamma correction before blitting to the swapchain image
		if color_space == ColorSpace::SrgbNonLinear {
			let image_extent = self.color_image.image().extent();
			let workgroups_x = image_extent[0].div_ceil(64);
			let layout = self.gamma_pipeline.layout().clone();
			cb.bind_pipeline_compute(self.gamma_pipeline.clone())?
				.bind_descriptor_sets(PipelineBindPoint::Compute, layout, 0, self.color_set.clone())?
				.dispatch([workgroups_x, image_extent[1], 1])?;
		}

		cb.blit_image(BlitImageInfo::images(self.color_image.image().clone(), swapchain_image))?;

		Ok(())
	}
}

// create the color and depth images
fn create_images(
	memory_allocator: Arc<StandardMemoryAllocator>,
	extent: [u32; 3],
	depth_format: Format,
) -> crate::Result<(Arc<ImageView>, Arc<ImageView>)>
{
	let color_create_info = ImageCreateInfo {
		format: Format::R16G16B16A16_SFLOAT,
		extent,
		usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
		..Default::default()
	};
	let color_image = Image::new(memory_allocator.clone(), color_create_info, AllocationCreateInfo::default())?;
	let color_image_view = ImageView::new_default(color_image)?;

	let depth_create_info = ImageCreateInfo {
		format: depth_format,
		extent,
		usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
		..Default::default()
	};
	let depth_image = Image::new(memory_allocator, depth_create_info, AllocationCreateInfo::default())?;
	let depth_image_view = ImageView::new_default(depth_image)?;

	Ok((color_image_view, depth_image_view))
}
