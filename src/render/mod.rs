/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod model;
pub mod pipeline;
pub mod skybox;
mod swapchain;
pub mod texture;
pub mod transparency;
mod vulkan_init;

use std::collections::HashMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use glam::*;

use vulkano::{Validated, VulkanError};
use vulkano::buffer::{
	allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
	subbuffer::Subbuffer,
	Buffer, BufferCreateInfo, BufferUsage,
};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	AutoCommandBufferBuilder, BlitImageInfo, CommandBufferInheritanceInfo,
	CommandBufferInheritanceRenderingInfo, CommandBufferUsage, CopyBufferInfo,
	CopyBufferToImageInfo, CopyImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
	SecondaryAutoCommandBuffer, SubpassContents, RenderingInfo, RenderingAttachmentInfo,
	CommandBufferExecFuture,
};
use vulkano::descriptor_set::{
	allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo}, DescriptorSet,
};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{view::ImageView, Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::{
	allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
	MemoryPropertyFlags,
};
use vulkano::pipeline::graphics::{viewport::Viewport, GraphicsPipeline};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::sync::{future::NowFuture, GpuFuture};
use winit::window::WindowBuilder;

use crate::GenericEngineError;
use pipeline::PipelineConfig;
use texture::Texture;

// Format used for main depth buffer.
// NOTE: While [NVIDIA recommends](https://developer.nvidia.com/blog/vulkan-dos-donts/) using a 24-bit depth format
// (`D24_UNORM_S8_UINT`), it doesn't seem to be very well-supported outside of NVIDIA GPUs. Only about 70% of GPUs on Windows
// and 50% of GPUs on Linux seem to support it, while `D16_UNORM` and `D32_SFLOAT` both have 100% support.
// (https://vulkan.gpuinfo.org/listoptimaltilingformats.php)
pub const MAIN_DEPTH_FORMAT: Format = Format::D16_UNORM;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	device: Arc<Device>,
	swapchain: swapchain::Swapchain,
	graphics_queue: Arc<Queue>,
	transfer_queue: Option<Arc<Queue>>, // if there is a separate (preferably dedicated) transfer queue, use it for transfers
	vram_all_host_visible: bool,
	descriptor_set_allocator: StandardDescriptorSetAllocator,
	memory_allocator: Arc<StandardMemoryAllocator>,
	command_buffer_allocator: StandardCommandBufferAllocator,

	cb_3d: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,

	// Futures from submitted immutable buffer/image transfers. Only used if a separate transfer queue exists.
	transfer_future: Option<CommandBufferExecFuture<NowFuture>>,

	// Loaded textures, with the key being the path relative to the current working directory
	textures: HashMap<PathBuf, Arc<Texture>>,

	// User-accessible material pipelines. Optional transparency pipeline may also be specified.
	material_pipelines: HashMap<String, (Arc<GraphicsPipeline>, Option<Arc<GraphicsPipeline>>)>,

	// The final contents of this render target's color image will be blitted to the intermediate sRGB nonlinear image.
	main_render_target: RenderTarget,

	// An sRGB image which the above `main_render_target` will be blitted to, thus converting it to nonlinear.
	// This will be copied, not blitted, to the swapchain.
	intermediate_srgb_img: Arc<Image>,

	transparency_renderer: transparency::MomentTransparencyRenderer,

	// A common `SubbufferAllocator` for high-frequency staging buffers.
	staging_buffer_allocator: Mutex<SubbufferAllocator>,

	// TODO: put non-material shaders (shadow filtering, post processing) into different containers
	last_frame_presented: std::time::Instant,
	frame_time: std::time::Duration,

	resize_this_frame: bool,

	// Transfers to submit on the asynchronous transfer queue.
	// These will be performed while the CPU is busy with building the draw command buffers.
	async_transfers: Vec<StagingWork>,

	// Transfers to submit on the graphics queue, before the next frame submission.
	transfers: Vec<StagingWork>,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) -> Result<Self, GenericEngineError>
	{
		let (graphics_queue, transfer_queue, vram_all_host_visible) = vulkan_init::vulkan_setup(game_name, event_loop)?;
		let vk_dev = graphics_queue.device().clone();

		let use_monitor = event_loop 
			.primary_monitor()
			.or(event_loop.available_monitors().next())
			.ok_or("The primary monitor could not be detected.")?;
		
		let current_video_mode = get_video_modes(use_monitor.clone());

		// attempt to use fullscreen window if requested
		// TODO: load this from config
		let fullscreen = if std::env::args().find(|arg| arg == "-fullscreen").is_some() {
			if std::env::args().find(|arg| arg == "-borderless").is_some() {
				// if "-fullscreen" and "-borderless" were both specified, make a borderless window filling the entire monitor
				// instead of exclusive fullscreen
				Some(winit::window::Fullscreen::Borderless(Some(use_monitor.clone())))
			} else {
				// NOTE: this is specifically *exclusive* fullscreen, which gets ignored on Wayland.
				// therefore, it might be a good idea to hide such an option in UI from the end user on Wayland.
				// TODO: use VK_EXT_full_screen_exclusive to minimize latency (usually only available on Windows)
				if let Some(fullscreen_video_mode) = current_video_mode {
					Some(winit::window::Fullscreen::Exclusive(fullscreen_video_mode))
				} else {
					log::warn!("The current monitor's video mode could not be determined. Fullscreen mode is unavailable.");
					None
				}
			}
		} else {
			None
		};

		let window_size = if let Some(fs_mode) = &fullscreen {
			match fs_mode {
				winit::window::Fullscreen::Exclusive(video_mode) => video_mode.size(),
				winit::window::Fullscreen::Borderless(_) => use_monitor.size()
			}
		} else {
			// TODO: load this from config
			winit::dpi::PhysicalSize::new(1280, 720)
		};

		// create window
		let window = WindowBuilder::new()
			.with_inner_size(window_size)
			.with_title(game_name)
			.with_decorations(std::env::args().find(|arg| arg == "-borderless").is_none())
			.with_fullscreen(fullscreen)
			.build(&event_loop)?;	
		
		let swapchain = swapchain::Swapchain::new(vk_dev.clone(), window)?;

		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
			vk_dev.clone(),
			StandardDescriptorSetAllocatorCreateInfo::default()
		);
		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_dev.clone()));

		// The counts below are multiplied by the number of swapchain images to account for previous submissions.
		let cb_alloc_info = StandardCommandBufferAllocatorCreateInfo {
			// One for graphics, another for async transfers, each of which are on separate queue families.
			primary_buffer_count: swapchain.image_count(),

			// Only one secondary command buffer should be created per thread.
			secondary_buffer_count: swapchain.image_count(),

			..Default::default()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(vk_dev.clone(), cb_alloc_info);

		let main_render_target = RenderTarget::new(memory_allocator.clone(), swapchain.dimensions())?;
		let swapchain_dim = swapchain.dimensions();
		let intermediate_img_create_info = ImageCreateInfo {
			format: Format::B8G8R8A8_SRGB,
			extent: [ swapchain_dim[0], swapchain_dim[1], 1 ],
			usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
			..Default::default()
		};
		let intermediate_srgb_img = 
			Image::new(memory_allocator.clone(), intermediate_img_create_info, AllocationCreateInfo::default())?;
		let transparency_renderer = transparency::MomentTransparencyRenderer::new(
			memory_allocator.clone(),
			&descriptor_set_allocator,
			swapchain.dimensions(),
		)?;

		let pool_create_info = SubbufferAllocatorCreateInfo {
			arena_size: 128 * 1024, // TODO: determine an appropriate arena size based on actual memory usage
			buffer_usage: BufferUsage::TRANSFER_SRC,
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buffer_allocator = Mutex::new(SubbufferAllocator::new(memory_allocator.clone(), pool_create_info));

		// TODO: adjust capacity of these based on number of components that might need to do a transfer
		let transfers = Vec::with_capacity(200);
		let async_transfers = Vec::with_capacity(200);

		Ok(RenderContext {
			device: vk_dev,
			swapchain,
			graphics_queue,
			transfer_queue,
			vram_all_host_visible,
			descriptor_set_allocator,
			memory_allocator,
			command_buffer_allocator,
			cb_3d: Mutex::new(None),
			transfer_future: None,
			textures: HashMap::new(),
			material_pipelines: HashMap::new(),
			main_render_target,
			intermediate_srgb_img,
			transparency_renderer,
			staging_buffer_allocator,
			last_frame_presented: std::time::Instant::now(),
			frame_time: std::time::Duration::ZERO,
			resize_this_frame: false,
			transfers,
			async_transfers,
		})
	}

	/// Load a material shader pipeline into memory, using a configuration.
	pub fn load_material_pipeline(&mut self, name: &str, mut config: PipelineConfig) -> Result<(), GenericEngineError>
	{
		let pipeline = pipeline::new_from_config(self.device.clone(), config.clone())?;

		let transparency_pipeline = if config.fragment_shader_transparency.is_some() {
			config.set_layouts.push(self.transparency_renderer.get_stage3_inputs().layout().clone());
			Some(pipeline::new_from_config_transparency(self.device.clone(), config)?)
		} else {
			None
		};

		self.material_pipelines.insert(
			name.to_string(),
			(pipeline, transparency_pipeline)
		);

		Ok(())
	}

	/// Load an image file as a texture into memory.
	/// If the image was already loaded, it'll use the corresponding texture.
	pub fn get_texture(&mut self, path: &Path) -> Result<Arc<Texture>, GenericEngineError>
	{
		match self.textures.get(path) {
			Some(tex) => {
				log::info!("Reusing loaded texture '{}'", path.display());
				Ok(tex.clone())
			}
			None => {
				let (tex, staging_work) = texture::Texture::new(self.memory_allocator.clone(), path)?;
				self.add_transfer(staging_work.into());
				let tex_arc = Arc::new(tex);
				self.textures.insert(path.to_path_buf(), tex_arc.clone());
				Ok(tex_arc)
			}
		}
	}

	pub fn new_cubemap_texture(&mut self, faces: [PathBuf; 6]) -> Result<texture::CubemapTexture, GenericEngineError>
	{
		let (tex, staging_work) = texture::CubemapTexture::new(self.memory_allocator.clone(), faces)?;
		self.add_transfer(staging_work.into());
		Ok(tex)
	}

	pub fn new_texture_from_iter<Px, I>(
		&mut self,
		iter: I,
		vk_fmt: Format,
		dimensions: [u32; 2],
		mip: u32,
	) -> Result<texture::Texture, GenericEngineError>
	where
		Px: Send + Sync + bytemuck::Pod,
		[Px]: vulkano::buffer::BufferContents,
		I: IntoIterator<Item = Px>,
		I::IntoIter: ExactSizeIterator,
	{
		let (tex, staging_work) = texture::Texture::new_from_iter(self.memory_allocator.clone(), iter, vk_fmt, dimensions, mip)?;
		self.add_transfer(staging_work.into());
		Ok(tex)
	}

	/// Create a device-local buffer from an iterator, initialized with `data` for `usage`.
	/// For stuff that isn't an array, just put the data into a single-element iterator, like `[data]`.
	pub fn new_buffer<I, T>(&mut self, data: I, usage: BufferUsage) -> Result<Subbuffer<[T]>, GenericEngineError>
	where
		T: Send + Sync + bytemuck::Pod,
		I: IntoIterator<Item = T>,
		I::IntoIter: ExactSizeIterator,
		[T]: vulkano::buffer::BufferContents,
	{
		let buf = if self.vram_all_host_visible {
			// If the entire VRAM is host-visible, upload directly to the new buffer.
			let buffer_info = BufferCreateInfo { usage, ..Default::default() };
			let staging_allocation_info = AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter {
					required_flags: MemoryPropertyFlags::DEVICE_LOCAL | MemoryPropertyFlags::HOST_VISIBLE,
					preferred_flags: MemoryPropertyFlags::empty(),
					not_preferred_flags: MemoryPropertyFlags::HOST_CACHED | MemoryPropertyFlags::DEVICE_COHERENT
						| MemoryPropertyFlags::DEVICE_UNCACHED,
				},
				..Default::default()
			};
			Buffer::from_iter(self.memory_allocator.clone(), buffer_info, staging_allocation_info, data)?
		} else {
			// If the above isn't true, create a staging buffer on the CPU side, 
			// then submit a transfer command to the new buffer on the GPU side.
			let buffer_info = BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() };
			let staging_alloc_info = AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			};
			let staging_buf = Buffer::from_iter(self.memory_allocator.clone(), buffer_info, staging_alloc_info, data)?;

			let buffer_info = BufferCreateInfo { usage: usage | BufferUsage::TRANSFER_DST, ..Default::default() };
			let new_buf = Buffer::new_slice(
				self.memory_allocator.clone(),
				buffer_info,
				AllocationCreateInfo::default(),
				staging_buf.len()
			)?;
			self.add_transfer(CopyBufferInfo::buffers(staging_buf, new_buf.clone()).into());
			new_buf
		};
		
		Ok(buf)
	}

	/// Get a staging buffer from the common staging buffer allocator, and queue an upload using it.
	pub fn copy_to_buffer<T>(&mut self, data: T, dst_buf: Subbuffer<T>) -> Result<(), GenericEngineError>
	where
		[T]: vulkano::buffer::BufferContents,
		T: Send + Sync + bytemuck::Pod,
	{
		let staging_buf = self
			.staging_buffer_allocator
			.lock()
			.or_else(|_| Err("Staging buffer allocator mutex is poisoned!"))?
			.allocate_sized()?;
		*staging_buf.write()? = data;
		self.add_transfer_on_graphics_queue(CopyBufferInfo::buffers(staging_buf, dst_buf).into());
		Ok(())
	}

	/// Add staging work for new objects.
	/// This may be performed asynchronously with the CPU building the draw command buffers,
	/// if there is a separate transfer queue.
	fn add_transfer(&mut self, work: StagingWork)
	{
		if self.transfer_queue.is_some() {
			if self.async_transfers.len() == self.async_transfers.capacity() {
				log::warn!(
					"Re-allocating `Vec` for asynchronous transfers to {}! Consider increasing its initial capacity.",
					self.async_transfers.len() + 1
				);
			}
			self.async_transfers.push(work);
		} else {
			self.add_transfer_on_graphics_queue(work);
		}
	}

	/// Submit a transfer on the graphics queue. Use this instead of `add_transfer` if
	/// there's the possibility that the destination object is in use by a previous submission.
	fn add_transfer_on_graphics_queue(&mut self, work: StagingWork)
	{
		if self.transfers.len() == self.transfers.capacity() {
			log::warn!(
				"Re-allocating `Vec` for transfers to {}! Consider increasing its initial capacity.",
				self.transfers.len() + 1
			);
		}
		self.transfers.push(work);
	}

	/// Submit the asynchronous transfers that are waiting.
	/// Run this just before beginning to build the draw command buffers,
	/// so that the transfers can be done while the CPU is busy with building the draw command buffers.
	pub fn submit_async_transfers(&mut self) -> Result<(), GenericEngineError>
	{
		if self.async_transfers.len() > 0 {
			if let Some(q) = self.transfer_queue.as_ref() {
				let mut cb = AutoCommandBufferBuilder::primary(
					&self.command_buffer_allocator,
					q.queue_family_index(),
					CommandBufferUsage::OneTimeSubmit,
				)?;

				for work in self.async_transfers.drain(..) {
					work.add_command(&mut cb)?;
				}

				let transfer_future = cb.build()?.execute(q.clone())?;
				transfer_future.flush()?;

				if self.transfer_future.is_some() {
					// Async transfer futures *must* be used when the draw commands are submitted,
					// otherwise we can't guarantee that the transfers have finished by the time
					// the draws that need them are performed.
					panic!("Unused async transfer future found!");
				}

				self.transfer_future = Some(transfer_future);
			} else {
				// This shouldn't be reached if the check for `transfer_queue`
				// being `Some` worked properly in `add_transfer`.
				unreachable!();
			}
		}

		Ok(())
	}

	/// Issue a new secondary command buffer builder to begin recording to.
	/// It will be set up for drawing to color and depth images with the given format,
	/// and with a viewport as large as `viewport_dimensions`.
	pub fn new_secondary_command_buffer(
		&self,
		color_attachment_formats: Vec<Option<Format>>,
		depth_attachment_format: Option<Format>,
		viewport_dimensions: [u32; 2]
	) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, Validated<VulkanError>>
	{
		let render_pass = Some(CommandBufferInheritanceRenderingInfo {
			color_attachment_formats,
			depth_attachment_format,
			..Default::default()
		}.into());
		let mut cb = AutoCommandBufferBuilder::secondary(
			&self.command_buffer_allocator,
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
			CommandBufferInheritanceInfo { render_pass, ..Default::default() },
		)?;

		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
			depth_range: 0.0..=1.0,
		};
		cb.set_viewport(0, [viewport].as_slice().into())?;

		Ok(cb)
	}

	fn resize_everything_else(&mut self) -> Result<(), GenericEngineError>
	{
		// Update images to match the current window size.
		self.main_render_target = RenderTarget::new(self.memory_allocator.clone(), self.swapchain.dimensions())?;

		let swapchain_dim = self.swapchain.dimensions();
		let intermediate_img_create_info = ImageCreateInfo {
			format: Format::B8G8R8A8_SRGB,
			extent: [ swapchain_dim[0], swapchain_dim[1], 1 ],
			usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
			..Default::default()
		};
		self.intermediate_srgb_img = 
			Image::new(self.memory_allocator.clone(), intermediate_img_create_info, AllocationCreateInfo::default())?;

		self.transparency_renderer.resize_image(
			self.memory_allocator.clone(),
			&self.descriptor_set_allocator,
			self.swapchain.dimensions()
		)?;

		Ok(())
	}

	pub fn resize_swapchain(&mut self) -> Result<(), GenericEngineError>
	{
		self.resize_this_frame = self.swapchain.fit_window()?;
		self.resize_everything_else()?;
		Ok(())
	}

	pub fn add_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		*self.cb_3d.lock().unwrap() = Some(cb);
	}

	/// Submit all the command buffers for this frame to actually render them to the image.
	pub fn submit_frame(&mut self, ui_cb: Option<Arc<SecondaryAutoCommandBuffer>>) -> Result<(), GenericEngineError>
	{
		let mut primary_cb_builder = AutoCommandBufferBuilder::primary(
			&self.command_buffer_allocator,
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
		)?;

		// transfers
		for work in self.transfers.drain(..) {
			work.add_command(&mut primary_cb_builder)?;
		}

		// 3D
		if let Some(some_cb_3d) = self.cb_3d.lock().unwrap().take() {
			primary_cb_builder
				.begin_rendering(self.main_render_target.first_rendering_info())?
				.execute_commands(some_cb_3d)?
				.end_rendering()?;
		}

		// 3D OIT
		self.transparency_renderer.process_transparency(
			&mut primary_cb_builder,
			self.main_render_target.color_image().clone(),
			self.main_render_target.depth_image().clone()
		)?;

		// UI
		if let Some(some_ui_cb) = ui_cb {
			let ui_render_info = RenderingInfo {
				color_attachments: vec![
					Some(RenderingAttachmentInfo {
						load_op: AttachmentLoadOp::Load,
						store_op: AttachmentStoreOp::Store,
						..RenderingAttachmentInfo::image_view(self.main_render_target.color_image().clone())
					}),
				],
				contents: SubpassContents::SecondaryCommandBuffers,
				..Default::default()
			};
			primary_cb_builder
				.begin_rendering(ui_render_info)?
				.execute_commands(some_ui_cb)?
				.end_rendering()?;
		}

		// blit to sRGB image to convert from linear to non-linear sRGB
		let blit_info = BlitImageInfo::images(
			self.main_render_target.color_image().image().clone(),
			self.intermediate_srgb_img.clone(),
		);
		primary_cb_builder.blit_image(blit_info)?;

		// get the next swapchain image (expected to be B8G8R8A8_UNORM), then copy the non-linear sRGB image to it
		let (swapchain_image, dimensions_changed) = self.swapchain.get_next_image()?;
		if dimensions_changed {
			self.resize_everything_else()?;
		}
		self.resize_this_frame = dimensions_changed;
		let copy_info = CopyImageInfo::images(self.intermediate_srgb_img.clone(), swapchain_image);
		primary_cb_builder.copy_image(copy_info)?;

		// finish building the command buffer, then present the swapchain image
		let transfer_future = self.transfer_future.take().map(|f| f.boxed_send_sync());
		self.swapchain.present(primary_cb_builder.build()?, self.graphics_queue.clone(), transfer_future)?;

		// set the delta time
		let now = std::time::Instant::now();
		let dur = now - self.last_frame_presented;
		self.last_frame_presented = now;
		self.frame_time = dur;

		Ok(())
	}

	pub fn descriptor_set_allocator(&self) -> &StandardDescriptorSetAllocator
	{
		&self.descriptor_set_allocator
	}

	/// Check if the window has been resized since the last frame submission.
	pub fn window_resized(&self) -> bool
	{
		self.resize_this_frame
	}

	pub fn swapchain_dimensions(&self) -> [u32; 2]
	{
		self.swapchain.dimensions()
	}

	pub fn get_transparency_renderer(&self) -> &transparency::MomentTransparencyRenderer
	{
		&self.transparency_renderer
	}

	pub fn get_pipeline(&self, name: &str) -> Result<&Arc<GraphicsPipeline>, PipelineNotLoaded>
	{
		Ok(self.material_pipelines.get(name).map(|tuple| &tuple.0).ok_or(PipelineNotLoaded)?)
	}
	pub fn get_transparency_pipeline(&self, name: &str) -> Result<&Arc<GraphicsPipeline>, PipelineNotLoaded>
	{
		Ok(self.material_pipelines.get(name).and_then(|tuple| tuple.1.as_ref()).ok_or(PipelineNotLoaded)?)
	}

	/// Get the delta time for last frame.
	pub fn delta(&self) -> std::time::Duration
	{
		self.frame_time
	}
}

// Get and print the video modes supported by the given monitor, and return the monitor's current video mode.
// This may return `None` if the monitor is currently using a video mode that it really doesn't support,
// although that should be rare.
fn get_video_modes(mon: winit::monitor::MonitorHandle) -> Option<winit::monitor::VideoMode>
{
	let mon_name = mon.name().unwrap_or_else(|| "[no longer exists]".to_string());

	let current_video_mode = mon
		.video_modes()
		.find(|vm| vm.size() == mon.size() && vm.refresh_rate_millihertz() == mon.refresh_rate_millihertz().unwrap_or(0));

	let mut video_modes: Vec<_> = mon.video_modes().collect();
	log::info!("All video modes supported by current monitor (\"{mon_name}\"):");

	// filter the video modes to those with >=1280 width, >=720 height, and not vertical (width <= height)
	video_modes.retain(|video_mode| {
		// print unfiltered video modes while we're at it
		let video_mode_suffix = match &current_video_mode {
			Some(cur_vm) if video_mode == cur_vm  => " <- likely current primary monitor video mode",
			_ => "",
		};
		log::info!("{}{}", format_video_mode(video_mode), video_mode_suffix);

		let size = video_mode.size();
		size.width >= 1280 && size.height >= 720 && size.width >= size.height
	});

	// filter the video modes to the highest refresh rate for each size
	// (sort beforehand so that highest refresh rate for each size comes first,
	// then remove duplicates with the same size and bit depth)
	video_modes.sort();
	video_modes.dedup_by_key(|video_mode| (video_mode.size(), video_mode.bit_depth()));

	log::info!("Filtered video modes:");
	for vm in &video_modes {
		log::info!("{}", format_video_mode(vm))
	}

	//(video_modes, current_video_mode)
	current_video_mode
}
fn format_video_mode(video_mode: &winit::monitor::VideoMode) -> String
{
	let size = video_mode.size();
	let refresh_rate_hz = video_mode.refresh_rate_millihertz() / 1000;
	let refresh_rate_thousandths = video_mode.refresh_rate_millihertz() % 1000;
	format!(
		"{} x {} @ {}.{:0>3} Hz {}-bit",
		size.width,
		size.height,
		refresh_rate_hz,
		refresh_rate_thousandths,
		video_mode.bit_depth()
	)
}

#[derive(Debug)]
pub struct PipelineNotLoaded;
impl std::error::Error for PipelineNotLoaded {}
impl std::fmt::Display for PipelineNotLoaded
{
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
	{
		write!(f, "the specified pipeline is not loaded")
	}
}

enum StagingWork
{
	CopyBuffer(CopyBufferInfo),
	CopyBufferToImage(CopyBufferToImageInfo),
}
impl StagingWork
{
	fn add_command(self, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
		-> Result<(), Box<vulkano::ValidationError>>
	{
		match self {
			StagingWork::CopyBuffer(info) => cb_builder.copy_buffer(info)?,
			StagingWork::CopyBufferToImage(info) => cb_builder.copy_buffer_to_image(info)?,
		};
		Ok(())
	}
}
impl From<CopyBufferInfo> for StagingWork
{
	fn from(info: CopyBufferInfo) -> StagingWork
	{
		Self::CopyBuffer(info)
	}
}
impl From<CopyBufferToImageInfo> for StagingWork
{
	fn from(info: CopyBufferToImageInfo) -> StagingWork
	{
		Self::CopyBufferToImage(info)
	}
}

struct RenderTarget
{
	color_image: Arc<ImageView>, // An FP16, linear gamma image which everything will be rendered to.
	depth_image: Arc<ImageView>,
}
impl RenderTarget
{
	pub fn new(memory_allocator: Arc<StandardMemoryAllocator>, dimensions: [u32; 2]) -> Result<Self, GenericEngineError>
	{
		let color_create_info = ImageCreateInfo {
			format: Format::R16G16B16A16_SFLOAT,
			extent: [ dimensions[0], dimensions[1], 1 ],
			usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
			..Default::default()
		};
		let color_image = Image::new(memory_allocator.clone(), color_create_info, AllocationCreateInfo::default())?;
		let color_image_view = ImageView::new_default(color_image)?;

		let depth_create_info = ImageCreateInfo {
			format: MAIN_DEPTH_FORMAT,
			extent: [ dimensions[0], dimensions[1], 1 ],
			usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
			..Default::default()
		};
		let depth_image = Image::new(memory_allocator, depth_create_info, AllocationCreateInfo::default())?;
		let depth_image_view = ImageView::new_default(depth_image)?;

		Ok(Self { 
			color_image: color_image_view, 
			depth_image: depth_image_view 
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

	pub fn first_rendering_info(&self) -> RenderingInfo
	{
		RenderingInfo {
			color_attachments: vec![
				Some(RenderingAttachmentInfo {
					// `load_op` default `DontCare` is used since drawing the skybox effectively clears the image for us
					store_op: AttachmentStoreOp::Store,
					..RenderingAttachmentInfo::image_view(self.color_image.clone())
				}),
			],
			depth_attachment: Some(RenderingAttachmentInfo{
				// `load_op` is `DontCare` here too since the skybox clears it with 1.0
				store_op: AttachmentStoreOp::Store, // order-independent transparency needs this to be `Store`
				..RenderingAttachmentInfo::image_view(self.depth_image.clone())
			}),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		}
	}
}
