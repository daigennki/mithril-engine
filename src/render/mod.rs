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
	Buffer, BufferContents, BufferCreateInfo, BufferUsage,
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
use vulkano::format::{ClearValue, Format};
use vulkano::image::{view::ImageView, Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::{
	allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
	MemoryPropertyFlags,
};
use vulkano::pipeline::graphics::{viewport::Viewport, GraphicsPipeline};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::sync::{future::{FenceSignalFuture, NowFuture}, GpuFuture};
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
	allow_direct_buffer_access: bool,
	descriptor_set_allocator: StandardDescriptorSetAllocator,
	memory_allocator: Arc<StandardMemoryAllocator>,
	command_buffer_allocator: StandardCommandBufferAllocator,

	cb_3d: Mutex<Option<Arc<SecondaryAutoCommandBuffer>>>,

	// Future from submitted immutable buffer/image transfers. Only used if a separate transfer queue exists.
	transfer_future: Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>,

	// Loaded textures, with the key being the path relative to the current working directory
	textures: HashMap<PathBuf, Arc<Texture>>,

	// User-accessible material pipelines. Optional transparency pipeline may also be specified.
	material_pipelines: HashMap<String, (Arc<GraphicsPipeline>, Option<Arc<GraphicsPipeline>>)>,

	main_render_target: RenderTarget,

	transparency_renderer: transparency::MomentTransparencyRenderer,

	// The subbuffer allocator for buffer updates.
	staging_buffer_allocator: Mutex<SubbufferAllocator>,
	staging_buf_max_size: usize, // Maximum staging buffer usage for the entire duration of the program.

	last_frame_presented: std::time::Instant,
	frame_time: std::time::Duration,

	resize_this_frame: bool,

	// Transfers to initialize buffers and images. If there is an asynchronous transfer queue,
	// these will be performed while the CPU is busy with building the draw command buffers.
	// Otherwise, these will be run at the beginning of the next graphics submission, before draws are performed.
	async_transfers: Vec<StagingWork>,

	// Buffer updates to run at the beginning of the next graphics submission.
	buffer_updates: Vec<Box<dyn UpdateBufferDataTrait>>,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) -> Result<Self, GenericEngineError>
	{
		let (graphics_queue, transfer_queue, allow_direct_buffer_access) = vulkan_init::vulkan_setup(game_name, event_loop)?;
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
		let transparency_renderer = transparency::MomentTransparencyRenderer::new(
			memory_allocator.clone(),
			&descriptor_set_allocator,
			swapchain.dimensions(),
		)?;

		let pool_create_info = SubbufferAllocatorCreateInfo {
			arena_size: 4096, // this should be adjusted based on actual memory usage
			buffer_usage: BufferUsage::TRANSFER_SRC,
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buffer_allocator = Mutex::new(SubbufferAllocator::new(memory_allocator.clone(), pool_create_info));

		// the capacity of these should be adjusted based on number of transfers that might be done in one frame
		let buffer_updates = Vec::with_capacity(16);
		let async_transfers = Vec::with_capacity(200);

		Ok(RenderContext {
			device: vk_dev,
			swapchain,
			graphics_queue,
			transfer_queue,
			allow_direct_buffer_access,
			descriptor_set_allocator,
			memory_allocator,
			command_buffer_allocator,
			cb_3d: Mutex::new(None),
			transfer_future: None,
			textures: HashMap::new(),
			material_pipelines: HashMap::new(),
			main_render_target,
			transparency_renderer,
			staging_buffer_allocator,
			staging_buf_max_size: 0,
			last_frame_presented: std::time::Instant::now(),
			frame_time: std::time::Duration::ZERO,
			resize_this_frame: false,
			buffer_updates,
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
		[Px]: BufferContents,
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
		T: Send + Sync + bytemuck::AnyBitPattern,
		I: IntoIterator<Item = T>,
		I::IntoIter: ExactSizeIterator,
		[T]: BufferContents,
	{
		let buf = if self.allow_direct_buffer_access {
			// When possible, upload directly to the new buffer memory.
			let buffer_info = BufferCreateInfo { usage, ..Default::default() };
			let alloc_info = AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter {
					required_flags: MemoryPropertyFlags::HOST_VISIBLE,
					preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL,
					not_preferred_flags: MemoryPropertyFlags::HOST_CACHED | MemoryPropertyFlags::DEVICE_COHERENT
						| MemoryPropertyFlags::DEVICE_UNCACHED,
				},
				..Default::default()
			};
			Buffer::from_iter(self.memory_allocator.clone(), buffer_info, alloc_info, data)?
		} else {
			// If direct uploads aren't possible, create a staging buffer on the CPU side, 
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

	/// Update a buffer at the begninning of the next graphics submission.
	pub fn update_buffer<T>(&mut self, data: T, dst_buf: Subbuffer<T>) -> Result<(), GenericEngineError>
	where
		T: BufferContents + Copy,
	{
		// This will be submitted to the graphics queue since we're copying to an existing buffer,
		// which might be in use by a previous submission.
		if self.buffer_updates.len() == self.buffer_updates.capacity() {
			log::warn!(
				"Re-allocating `Vec` for buffer updates to {}! Consider increasing its initial capacity.",
				self.buffer_updates.len() + 1
			);
		}
		self.buffer_updates.push(Box::new(UpdateBufferData { dst_buf, data }));

		Ok(())
	}

	/// Add staging work for new objects.
	/// If there is a separate transfer queue, this will be performed asynchronously with the CPU
	/// building the draw command buffers.
	fn add_transfer(&mut self, work: StagingWork)
	{
		if self.async_transfers.len() == self.async_transfers.capacity() {
			log::warn!(
				"Re-allocating `Vec` for asynchronous transfers to {}! Consider increasing its initial capacity.",
				self.async_transfers.len() + 1
			);
		}
		self.async_transfers.push(work);
	}

	/// Submit the asynchronous transfers that are waiting.
	/// Run this just before beginning to build the draw command buffers,
	/// so that the transfers can be done while the CPU is busy with building the draw command buffers.
	///
	/// This does nothing if there is no asynchronous transfer queue. In such a case, the transfers will
	/// instead be done at the beginning of the graphics submission on the graphics queue.
	pub fn submit_async_transfers(&mut self) -> Result<(), GenericEngineError>
	{
		if let Some(q) = self.transfer_queue.as_ref() {
			if self.async_transfers.len() > 0 {
				let mut cb = AutoCommandBufferBuilder::primary(
					&self.command_buffer_allocator,
					q.queue_family_index(),
					CommandBufferUsage::OneTimeSubmit,
				)?;

				for work in self.async_transfers.drain(..) {
					work.add_command(&mut cb)?;
				}

				let transfer_future = cb
					.build()?
					.execute(q.clone())?
					.then_signal_fence_and_flush()?;

				// This panics here if there's an unused future, because it *must* have been used when
				// the draw commands were submitted last frame. Otherwise, we can't guarantee that transfers
				// have finished by the time the draws that need them are performed.
				assert!(self.transfer_future.replace(transfer_future).is_none());
			}
		}
		
		Ok(())
	}

	/// Issue a new secondary command buffer builder to begin recording to.
	/// It will be set up for drawing to color and depth images with the given format,
	/// and with a viewport as large as `viewport_extent`.
	pub fn gather_commands(
		&self,
		color_attachment_formats: &[Format],
		depth_attachment_format: Option<Format>,
		viewport_extent: [u32; 2]
	) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, Validated<VulkanError>>
	{
		let render_pass = Some(CommandBufferInheritanceRenderingInfo {
			color_attachment_formats: color_attachment_formats.iter().map(|f| Some(*f)).collect(),
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
			extent: [viewport_extent[0] as f32, viewport_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};
		cb.set_viewport(0, [viewport].as_slice().into())?;

		Ok(cb)
	}

	fn resize_everything_else(&mut self) -> Result<(), GenericEngineError>
	{
		// Update images to match the current window size.
		self.main_render_target = RenderTarget::new(self.memory_allocator.clone(), self.swapchain.dimensions())?;
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
	pub fn submit_frame(
		&mut self,
		ui_cb: Option<Arc<SecondaryAutoCommandBuffer>>, 
		dir_light_shadows: Vec<(Arc<SecondaryAutoCommandBuffer>, Arc<ImageView>)>,
	) -> Result<(), GenericEngineError>
	{
		let mut primary_cb_builder = AutoCommandBufferBuilder::primary(
			&self.command_buffer_allocator,
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
		)?;

		// buffer updates
		if self.buffer_updates.len() > 0 {
			// this `Mutex` should never be poisoned since it's only used here
			let mut staging_buf_alloc_guard = self.staging_buffer_allocator.lock().unwrap();
			let mut staging_buf_usage_frame = 0;

			for buf_update in self.buffer_updates.drain(..) {
				buf_update.add_command(&mut primary_cb_builder, &mut staging_buf_alloc_guard)?;
				staging_buf_usage_frame += buf_update.data_size();
			}

			// gather stats on staging buffer usage
			if staging_buf_usage_frame > self.staging_buf_max_size {
				self.staging_buf_max_size = staging_buf_usage_frame;
				log::debug!("max staging buffer usage per frame: {} bytes", self.staging_buf_max_size);
			}
		}

		// do async transfers that couldn't be submitted earlier
		for work in self.async_transfers.drain(..) {
			work.add_command(&mut primary_cb_builder)?;
		}

		// shadows
		for (shadow_cb, shadow_layer_image_view) in dir_light_shadows {
			let shadow_render_info = RenderingInfo {
				depth_attachment: Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Depth(1.0)),
					..RenderingAttachmentInfo::image_view(shadow_layer_image_view)
				}),
				contents: SubpassContents::SecondaryCommandBuffers,
				..Default::default()
			};
			primary_cb_builder
				.begin_rendering(shadow_render_info)?
				.execute_commands(shadow_cb)?
				.end_rendering()?;
		}

		// get the next swapchain image (expected to be B8G8R8A8_UNORM)
		let (swapchain_image, dimensions_changed) = self.swapchain.get_next_image()?;
		if dimensions_changed {
			self.resize_everything_else()?;
		}
		self.resize_this_frame = dimensions_changed;

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

		// copy the non-linear sRGB image to the swapchain image
		self.main_render_target.copy_to_swapchain(&mut primary_cb_builder, swapchain_image)?;

		// finish building the command buffer, then present the swapchain image
		let transfer_future = self.transfer_future.take();
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

	pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator>
	{
		&self.memory_allocator
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

struct UpdateBufferData<T: BufferContents + Copy>
{
	pub dst_buf: Subbuffer<T>,
	pub data: T,
}
impl<T: BufferContents + Copy> UpdateBufferDataTrait for UpdateBufferData<T>
{
	fn data_size(&self) -> usize
	{
		std::mem::size_of::<T>()
	}

	fn add_command(
		&self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		subbuffer_allocator: &mut SubbufferAllocator
	) -> Result<(), GenericEngineError>
	{
		let staging_buf = subbuffer_allocator.allocate_sized()?;
		*staging_buf.write()? = self.data.clone();

		// TODO: actually use `update_buffer` when the `'static` requirement gets removed for the data
		cb_builder.copy_buffer(CopyBufferInfo::buffers(staging_buf, self.dst_buf.clone()))?;

		Ok(())
	}
}
trait UpdateBufferDataTrait: Send + Sync
{
	fn data_size(&self) -> usize;

	fn add_command(
		&self,
		_: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, 
		_: &mut SubbufferAllocator
	) -> Result<(), GenericEngineError>;
}

enum StagingWork
{
	CopyBuffer(CopyBufferInfo),
	CopyBufferToImage(CopyBufferToImageInfo),
}
impl StagingWork
{
	fn add_command(self, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
		-> Result<(), GenericEngineError>
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

	// An sRGB image which the above `color_image` will be blitted to, thus converting it to nonlinear.
	// This will be copied, not blitted, to the swapchain.
	srgb_image: Arc<Image>,
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
		let depth_image = Image::new(memory_allocator.clone(), depth_create_info, AllocationCreateInfo::default())?;
		let depth_image_view = ImageView::new_default(depth_image)?;

		let srgb_img_create_info = ImageCreateInfo {
			format: Format::B8G8R8A8_SRGB,
			extent: [ dimensions[0], dimensions[1], 1 ],
			usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
			..Default::default()
		};
		let srgb_image = Image::new(memory_allocator, srgb_img_create_info, AllocationCreateInfo::default())?;

		Ok(Self { 
			color_image: color_image_view, 
			depth_image: depth_image_view,
			srgb_image,
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
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store, // order-independent transparency needs this to be `Store`
				clear_value: Some(ClearValue::Depth(1.0)),
				..RenderingAttachmentInfo::image_view(self.depth_image.clone())
			}),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		}
	}

	pub fn copy_to_swapchain(
		&self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		swapchain_image: Arc<Image>
	) -> Result<(), GenericEngineError>
	{
		// blit to sRGB image to convert from linear to non-linear sRGB,
		// then copy the non-linear image to the swapchain
		cb
			.blit_image(BlitImageInfo::images(self.color_image.image().clone(), self.srgb_image.clone()))?
			.copy_image(CopyImageInfo::images(self.srgb_image.clone(), swapchain_image))?;

		Ok(())
	}
}
