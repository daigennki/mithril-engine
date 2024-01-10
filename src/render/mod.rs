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
pub mod ui;
mod vulkan_init;
pub mod workload;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use glam::*;

use vulkano::buffer::{subbuffer::Subbuffer, Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{
	allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo};
use vulkano::memory::{
	allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
	MemoryPropertyFlags,
};
use vulkano::pipeline::graphics::depth_stencil::CompareOp;
use vulkano::shader::ShaderStages;

use texture::Texture;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	swapchain: swapchain::Swapchain,
	main_render_target: render_target::RenderTarget,
	memory_allocator: Arc<StandardMemoryAllocator>,
	descriptor_set_allocator: StandardDescriptorSetAllocator,
	command_buffer_allocator: StandardCommandBufferAllocator,

	transparency_renderer: Option<transparency::MomentTransparencyRenderer>,

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

		let main_render_target = render_target::RenderTarget::new(
			memory_allocator.clone(),
			&descriptor_set_allocator,
			swapchain.get_images(),
			swapchain.color_space(),
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
			transparency_renderer: None,
			light_set_layout,
			textures: HashMap::new(),
			allow_direct_buffer_access,
			transfer_manager,
		})
	}

	fn load_transparency(&mut self, material_textures_set_layout: Arc<DescriptorSetLayout>) -> crate::Result<()>
	{
		self.transparency_renderer = Some(transparency::MomentTransparencyRenderer::new(
			self.memory_allocator.clone(),
			&self.descriptor_set_allocator,
			material_textures_set_layout,
			self.swapchain.dimensions(),
			self.depth_stencil_format(),
		)?);
		Ok(())
	}

	/// Create a device-local buffer from a slice, initialized with `data` for `usage`.
	/// For stuff that isn't an array, just put the data into a single-element slice, like `[data]`.
	fn new_buffer<T>(&mut self, data: Vec<T>, usage: BufferUsage) -> crate::Result<Subbuffer<[T]>>
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
			buf.write().unwrap().copy_from_slice(&data);
		} else {
			// If direct uploads aren't possible, create a staging buffer on the CPU side,
			// then submit a transfer command to the new buffer on the GPU side.
			log::debug!("Allocating buffer of {} bytes", data_size_bytes);

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
			self.transfer_manager.copy_to_buffer(data, buf.clone());
		}
		Ok(buf)
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
		if let Some(transparency_renderer) = &mut self.transparency_renderer {
			transparency_renderer.resize_image(
				self.memory_allocator.clone(),
				&self.descriptor_set_allocator,
				self.swapchain.dimensions(),
			)?
		}

		Ok(())
	}

	fn depth_stencil_format(&self) -> Format
	{
		self.main_render_target.depth_image().format()
	}

	fn submit_primary(&mut self, built_cb: Arc<PrimaryAutoCommandBuffer>) -> crate::Result<()>
	{
		self.swapchain.submit(built_cb, self.transfer_manager.take_transfer_future())
	}

	fn graphics_queue_family_index(&self) -> u32
	{
		self.swapchain.graphics_queue_family_index()
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
