/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod lighting;
pub mod model;
pub mod pipeline;
mod render_target;
pub mod skybox;
mod swapchain;
pub mod texture;
mod transfer;
mod transparency;
mod vulkan_init;
pub mod workload;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use glam::*;

use vulkano::buffer::{subbuffer::Subbuffer, Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	CopyBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{
	allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
	layout::{
		DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
	},
	DescriptorSet,
};
use vulkano::device::{Device, DeviceOwned};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo};
use vulkano::memory::{
	allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
	MemoryPropertyFlags,
};
use vulkano::pipeline::graphics::depth_stencil::CompareOp;
use vulkano::shader::ShaderStages;

use texture::Texture;

// Format used for main depth buffer.
//
// NOTE: While [NVIDIA recommends](https://developer.nvidia.com/blog/vulkan-dos-donts/) using a
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
pub const MAIN_DEPTH_FORMAT: Format = Format::D16_UNORM;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	swapchain: swapchain::Swapchain,
	main_render_target: render_target::RenderTarget,
	memory_allocator: Arc<StandardMemoryAllocator>,
	descriptor_set_allocator: StandardDescriptorSetAllocator,
	command_buffer_allocator: StandardCommandBufferAllocator,

	transparency_renderer: transparency::MomentTransparencyRenderer,

	material_textures_set_layout: Arc<DescriptorSetLayout>,
	light_set_layout: Arc<DescriptorSetLayout>,

	// Loaded textures, with the key being the path relative to the current working directory
	textures: HashMap<PathBuf, Arc<Texture>>,

	allow_direct_buffer_access: bool,

	transfer_manager: transfer::TransferManager,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) -> crate::Result<Self>
	{
		let (graphics_queue, transfer_queue, allow_direct_buffer_access) = vulkan_init::vulkan_setup(game_name, event_loop)?;
		let vk_dev = graphics_queue.device().clone();

		let swapchain = swapchain::Swapchain::new(graphics_queue, event_loop, game_name)?;

		let descriptor_set_allocator =
			StandardDescriptorSetAllocator::new(vk_dev.clone(), StandardDescriptorSetAllocatorCreateInfo::default());
		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_dev.clone()));

		// The counts below are multiplied by the number of swapchain images, to account for previous submissions.
		// - Primary: One for graphics, another for async transfers, each on separate queue families.
		// - Secondary: Only up to four should be created per thread.
		let cb_alloc_info = StandardCommandBufferAllocatorCreateInfo {
			primary_buffer_count: swapchain.image_count(),
			secondary_buffer_count: 4 * swapchain.image_count(),
			..Default::default()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(vk_dev.clone(), cb_alloc_info);

		/* Common material texture descriptor set layout */
		let mat_tex_sampler_info = SamplerCreateInfo {
			anisotropy: Some(16.0),
			..SamplerCreateInfo::simple_repeat_linear()
		};
		let mat_tex_sampler = Sampler::new(vk_dev.clone(), mat_tex_sampler_info)?;
		let mat_tex_bindings = [
			DescriptorSetLayoutBinding {
				// binding 0: sampler0
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![mat_tex_sampler],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
			},
			DescriptorSetLayoutBinding {
				// binding 1: textures
				binding_flags: DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
				descriptor_count: 32,
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
			},
		];
		let mat_tex_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: (0..).zip(mat_tex_bindings).collect(),
			..Default::default()
		};
		let material_textures_set_layout = DescriptorSetLayout::new(vk_dev.clone(), mat_tex_set_layout_info)?;

		let main_render_target = render_target::RenderTarget::new(
			memory_allocator.clone(),
			&descriptor_set_allocator,
			swapchain.get_images(),
			swapchain.color_space(),
		)?;
		let transparency_renderer = transparency::MomentTransparencyRenderer::new(
			memory_allocator.clone(),
			&descriptor_set_allocator,
			material_textures_set_layout.clone(),
			swapchain.dimensions(),
		)?;

		/* descriptor set with everything lighting- and shadow-related */
		let shadow_sampler_info = SamplerCreateInfo {
			mag_filter: Filter::Linear,
			min_filter: Filter::Linear,
			compare: Some(CompareOp::LessOrEqual),
			..Default::default()
		};
		let shadow_sampler = Sampler::new(vk_dev.clone(), shadow_sampler_info)?;
		let light_bindings = [
			DescriptorSetLayoutBinding {
				// binding 0: shadow sampler
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![shadow_sampler],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
			},
			DescriptorSetLayoutBinding {
				// binding 1: directional light buffer
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
			},
			DescriptorSetLayoutBinding {
				// binding 2: directional light shadow
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
			},
		];
		let light_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: (0..).zip(light_bindings).collect(),
			..Default::default()
		};
		let light_set_layout = DescriptorSetLayout::new(vk_dev.clone(), light_set_layout_info)?;

		let transfer_manager = transfer::TransferManager::new(transfer_queue, memory_allocator.clone());

		Ok(RenderContext {
			swapchain,
			memory_allocator,
			descriptor_set_allocator,
			command_buffer_allocator,
			main_render_target,
			transparency_renderer,
			material_textures_set_layout,
			light_set_layout,
			textures: HashMap::new(),
			allow_direct_buffer_access,
			transfer_manager,
		})
	}

	/// Create a device-local buffer from a slice, initialized with `data` for `usage`.
	/// For stuff that isn't an array, just put the data into a single-element slice, like `[data]`.
	pub fn new_buffer<T>(&mut self, data: &[T], usage: BufferUsage) -> crate::Result<Subbuffer<[T]>>
	where
		T: BufferContents + Copy,
	{
		let data_len = data.len().try_into().unwrap();
		let data_size_bytes = data.len() * std::mem::size_of::<T>();
		let buf;
		if self.allow_direct_buffer_access {
			log::debug!("Allocating direct buffer of {} bytes", data_size_bytes);
			// When possible, upload directly to the new buffer memory.
			let buf_info = BufferCreateInfo {
				usage,
				..Default::default()
			};
			let alloc_info = AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter {
					required_flags: MemoryPropertyFlags::HOST_VISIBLE,
					preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL,
					not_preferred_flags: MemoryPropertyFlags::HOST_CACHED
						| MemoryPropertyFlags::DEVICE_COHERENT
						| MemoryPropertyFlags::DEVICE_UNCACHED,
				},
				..Default::default()
			};
			buf = Buffer::new_slice(self.memory_allocator.clone(), buf_info, alloc_info, data_len)?;
			buf.write().unwrap().copy_from_slice(data);
		} else {
			log::debug!("Allocating buffer of {} bytes", data_size_bytes);
			// If direct uploads aren't possible, create a staging buffer on the CPU side,
			// then submit a transfer command to the new buffer on the GPU side.
			let staging_buf = self.transfer_manager.get_staging_buffer(data.len().try_into().unwrap())?;
			staging_buf.write().unwrap().copy_from_slice(data);

			let buf_info = BufferCreateInfo {
				usage: usage | BufferUsage::TRANSFER_DST,
				..Default::default()
			};
			buf = Buffer::new_slice(
				self.memory_allocator.clone(),
				buf_info,
				AllocationCreateInfo::default(),
				data_len,
			)?;

			self.transfer_manager
				.add_transfer(CopyBufferInfo::buffers(staging_buf, buf.clone()).into());
		}
		Ok(buf)
	}

	/// Update a buffer at the begninning of the next graphics submission.
	pub fn update_buffer<T>(&mut self, data: &[T], dst_buf: Subbuffer<[T]>)
	where
		T: BufferContents + Copy,
	{
		self.transfer_manager.update_buffer(data, dst_buf);
	}

	fn submit_async_transfers(&mut self) -> crate::Result<()>
	{
		self.transfer_manager.submit_async_transfers(&self.command_buffer_allocator)
	}

	fn resize_everything_else(&mut self) -> crate::Result<()>
	{
		// Update images to match the current swapchain image extent.
		self.main_render_target.resize(
			self.memory_allocator.clone(),
			&self.descriptor_set_allocator,
			self.swapchain.get_images(),
			self.swapchain.color_space(),
		)?;
		self.transparency_renderer.resize_image(
			self.memory_allocator.clone(),
			&self.descriptor_set_allocator,
			self.swapchain.dimensions(),
		)?;

		Ok(())
	}

	fn submit_primary(&mut self, built_cb: Arc<PrimaryAutoCommandBuffer>) -> crate::Result<()>
	{
		self.swapchain.submit(built_cb, self.transfer_manager.take_transfer_future())
	}

	pub fn get_material_textures_set_layout(&self) -> &Arc<DescriptorSetLayout>
	{
		&self.material_textures_set_layout
	}
	pub fn get_light_set_layout(&self) -> &Arc<DescriptorSetLayout>
	{
		&self.light_set_layout
	}
	pub fn get_oit_stage3_input_layout(&self) -> &Arc<DescriptorSetLayout>
	{
		&self.transparency_renderer.get_stage3_inputs().layout()
	}

	pub fn device(&self) -> &Arc<Device>
	{
		self.memory_allocator.device()
	}
	pub fn graphics_queue_family_index(&self) -> u32
	{
		self.swapchain.graphics_queue_family_index()
	}

	pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator>
	{
		&self.memory_allocator
	}
	pub fn descriptor_set_allocator(&self) -> &StandardDescriptorSetAllocator
	{
		&self.descriptor_set_allocator
	}
	pub fn command_buffer_allocator(&self) -> &StandardCommandBufferAllocator
	{
		&self.command_buffer_allocator
	}

	/// Check if the window has been resized since the last frame submission.
	pub fn window_resized(&self) -> bool
	{
		self.swapchain.extent_changed()
	}

	pub fn set_fullscreen(&self, fullscreen: bool)
	{
		self.swapchain.set_fullscreen(fullscreen)
	}
	pub fn is_fullscreen(&self) -> bool
	{
		self.swapchain.is_fullscreen()
	}

	pub fn swapchain_dimensions(&self) -> [u32; 2]
	{
		self.swapchain.dimensions()
	}

	/// Get the delta time for last frame.
	pub fn delta(&self) -> std::time::Duration
	{
		self.swapchain.delta()
	}
}
