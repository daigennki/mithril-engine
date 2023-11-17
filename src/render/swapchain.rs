/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;
use vulkano::command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{Image, ImageUsage};
use vulkano::swapchain::{
	FullScreenExclusive, PresentMode, 
	Surface, SurfaceInfo, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::{
	future::{FenceSignalFuture, GpuFuture, NowFuture}
};
use winit::window::{Window, WindowBuilder};
use winit::monitor::{MonitorHandle, VideoMode};

#[cfg(target_family = "windows")]
use winit::platform::windows::MonitorHandleExtWindows;

use crate::GenericEngineError;

pub struct Swapchain
{
	window: Arc<Window>,
	swapchain: Arc<vulkano::swapchain::Swapchain>,
	images: Vec<Arc<Image>>,

	extent_changed: bool, // `true` if image extent changed since the last presentation

	acquire_future: Option<SwapchainAcquireFuture>,
	submission_future: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,

	last_frame_presented: std::time::Instant,
	frame_time: std::time::Duration,
}
impl Swapchain
{
	pub fn new(
		vk_dev: Arc<Device>, 
		event_loop: &winit::event_loop::EventLoop<()>,
		window_title: &str
	) -> Result<Self, GenericEngineError>
	{
		let use_monitor = event_loop 
			.primary_monitor()
			.or(event_loop.available_monitors().next())
			.ok_or("The primary monitor could not be detected.")?;
		
		let (_current_video_mode, fullscreen_mode) = get_video_modes(use_monitor.clone());

		let window_size = if let Some(fs_mode) = &fullscreen_mode {
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
			.with_title(window_title)
			.with_decorations(std::env::args().find(|arg| arg == "-borderless").is_none())
			.with_fullscreen(fullscreen_mode)
			.build(&event_loop)?;
		let window_arc = Arc::new(window);

		let surface = Surface::from_window(vk_dev.instance().clone(), window_arc.clone())?;

		// Enable exclusive fullscreen using VK_EXT_full_screen_exclusive when possible
		let surface_info = if vk_dev.enabled_extensions().ext_full_screen_exclusive {
			#[cfg(target_family = "windows")]
			let win32_monitor = Some(unsafe { vulkano::swapchain::Win32Monitor::new(use_monitor.hmonitor() as *const ()) });
			#[cfg(not(target_family = "windows"))]
			let win32_monitor = None;
			SurfaceInfo {
				full_screen_exclusive: FullScreenExclusive::Allowed,
				win32_monitor,
				..Default::default()
			}
		} else {
			SurfaceInfo::default()
		};

		let pd = vk_dev.physical_device();
		let surface_caps = pd.surface_capabilities(&surface, surface_info.clone())?;
		let surface_formats = pd.surface_formats(&surface, SurfaceInfo::default())?;
		let surface_present_modes = pd.surface_present_modes(&surface, SurfaceInfo::default())?;
		
		if vk_dev.enabled_extensions().ext_full_screen_exclusive {
			let full_screen_exclusive_supported_str = surface_caps
				.full_screen_exclusive_supported
				.then_some("available")
				.unwrap_or("not available");
			log::info!("Exclusive full-screen is {} for the primary monitor.", full_screen_exclusive_supported_str);
		}

		log::info!("Available surface format and color space combinations:");
		surface_formats.iter().for_each(|f| log::info!("{:?}", f));

		log::info!("Available surface present modes: {:?}", Vec::from_iter(surface_present_modes));

		// NVIDIA on Linux (possibly only when using Wayland with PRIME?) only supports B8G8R8A8_UNORM + SrgbNonLinear, so it
		// would be a safer bet than B8G8R8A8_SRGB. B8G8R8A8_UNORM does in fact have slightly wider support than B8G8R8A8_SRGB:
		// https://vulkan.gpuinfo.org/listsurfaceformats.php?platform=linux
		// This means we must convert to non-linear sRGB beforehand. See `RenderTarget::copy_to_swapchain` for that conversion.
		let create_info = SwapchainCreateInfo {
			min_image_count: surface_caps.min_image_count,
			image_extent: window_size.into(),
			image_format: Format::B8G8R8A8_UNORM,
			image_usage: ImageUsage::TRANSFER_DST,
			present_mode: PresentMode::Fifo,
			full_screen_exclusive: surface_info.full_screen_exclusive,
			win32_monitor: surface_info.win32_monitor,
			..Default::default()
		};

		let (swapchain, images) = vulkano::swapchain::Swapchain::new(vk_dev.clone(), surface, create_info)?;
		log::debug!("created {} swapchain images", images.len());

		Ok(Swapchain {
			window: window_arc,
			swapchain,
			images,
			extent_changed: false,
			acquire_future: None,
			submission_future: None,
			last_frame_presented: std::time::Instant::now(),
			frame_time: std::time::Duration::ZERO,
		})
	}

	/// Get the next swapchain image.
	pub fn get_next_image(&mut self) -> Result<Arc<Image>, GenericEngineError>
	{
		// Panic if this function is called when an image has already been acquired without being submitted
		assert!(self.acquire_future.is_none());

		// clean up resources from finished submissions
		if let Some(f) = self.submission_future.as_mut() {
			f.cleanup_finished();
		}

		// Recreate the swapchain if the surface's properties changed (e.g. window size changed).
		let new_inner_size = self.window.inner_size().into();
		if self.swapchain.image_extent() != new_inner_size {
			log::info!(
				"Window resized from {:?} to {:?}, recreating swapchain...",
				self.swapchain.image_extent(),
				new_inner_size
			);

			// set minimum size here to make sure we adapt to any DPI scale factor changes that may arise
			self.window.set_min_inner_size(Some(winit::dpi::PhysicalSize::new(1280, 720)));

			let create_info = SwapchainCreateInfo {
				image_extent: new_inner_size,
				..self.swapchain.create_info()
			};

			let (new_swapchain, new_images) = self.swapchain.recreate(create_info)?;
			self.swapchain = new_swapchain;
			self.images = new_images;

			self.extent_changed = true;
		} else {
			self.extent_changed = false;
		}

		let (image_num, suboptimal, acquire_future) = vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None)?;
		if suboptimal {
			log::warn!("Swapchain is suboptimal!");
		}
		self.acquire_future = Some(acquire_future);

		Ok(self.images[image_num as usize].clone())
	}

	/// Submit a primary command buffer's commands (where the command buffer is expected to manipulate the currently acquired
	/// swapchain image, usually blitting to it) and then present the resulting image.
	/// Optionally, a future `after` to wait for (usually for joining submitted transfers on another queue) can be given, so
	/// that graphics operations don't begin until after that future is reached.
	pub fn present(
		&mut self,
		cb: Arc<PrimaryAutoCommandBuffer>,
		queue: Arc<Queue>,
		after: Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>,
	) -> Result<(), GenericEngineError>
	{
		let acquire_future = self
			.acquire_future
			.take()
			.expect("Command buffer submit attempted without acquiring an image!");

		let present_info = SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), acquire_future.image_index());

		let mut joined_futures = acquire_future.boxed_send_sync();

		if let Some(f) = self.submission_future.take() {
			f.wait(None)?; // wait for the previous submission to finish, to make sure resources are no longer in use
			joined_futures = Box::new(joined_futures.join(f));
		}

		if let Some(f) = after {
			// Ideally we'd use a semaphore instead of a fence here, but apprently it's borked in Vulkano right now.
			f.wait(None)?;
			joined_futures = Box::new(joined_futures.join(f));
		}

		let submission_future = joined_futures
			.then_execute(queue.clone(), cb)?
			.then_swapchain_present(queue, present_info)
			.boxed_send_sync()
			.then_signal_fence_and_flush()?;
		self.submission_future = Some(submission_future);

		self.calculate_delta();

		Ok(())
	}

	/// Submit a command buffer without presenting a swapchain image.
	/// Only call this if `present` wasn't called for a frame, which may be the case when the window is minimized.
	pub fn submit_without_present(
		&mut self,
		cb: Arc<PrimaryAutoCommandBuffer>,
		queue: Arc<Queue>,
		after: Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>,
	) -> Result<(), GenericEngineError>
	{
		let mut joined_futures = vulkano::sync::future::now(queue.device().clone()).boxed_send_sync();

		if let Some(f) = self.submission_future.take() {
			f.wait(None)?; // wait for the previous submission to finish, to make sure resources are no longer in use
			joined_futures = Box::new(joined_futures.join(f));
		}

		if let Some(f) = after {
			// Ideally we'd use a semaphore instead of a fence here, but apprently it's borked in Vulkano right now.
			f.wait(None)?;
			joined_futures = Box::new(joined_futures.join(f));
		}

		let submission_future = joined_futures
			.then_execute(queue.clone(), cb)?
			.boxed_send_sync()
			.then_signal_fence_and_flush()?;
		self.submission_future = Some(submission_future);

		self.calculate_delta();

		Ok(())
	}

	fn calculate_delta(&mut self)
	{
		let now = std::time::Instant::now();
		let dur = now - self.last_frame_presented;
		self.last_frame_presented = now;
		self.frame_time = dur;
	}

	pub fn image_count(&self) -> usize
	{
		self.images.len()
	}

	pub fn dimensions(&self) -> [u32; 2]
	{
		self.swapchain.image_extent()
	}

	/// Check whether or not the image extent of the swapchain chaged since the last image was presented.
	pub fn extent_changed(&self) -> bool
	{
		self.extent_changed
	}

	/// Check if the window is currently minimized.
	pub fn window_minimized(&self) -> bool
	{
		self.window.is_minimized().unwrap_or(false)
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
fn get_video_modes(mon: MonitorHandle) -> (Option<VideoMode>, Option<winit::window::Fullscreen>)
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

	// attempt to use fullscreen window if requested
	// TODO: load this from config
	let fullscreen_mode = get_fullscreen_mode(mon, &current_video_mode);

	//(video_modes, current_video_mode)
	(current_video_mode, fullscreen_mode)
}
fn format_video_mode(video_mode: &VideoMode) -> String
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

// Determine the appropriate fullscreen mode depending on the arguments given to the executable.
// Returns `Some` if some kind of fullscreen mode is enabled, or `None` if the window should simply be in windowed mode.
fn get_fullscreen_mode(use_monitor: MonitorHandle, current_video_mode: &Option<VideoMode>) -> Option<winit::window::Fullscreen>
{
	if std::env::args().find(|arg| arg == "-fullscreen").is_some() {
		if std::env::args().find(|arg| arg == "-borderless").is_some() {
			// If "-fullscreen" and "-borderless" were both specified, make a borderless window filling the entire monitor
			// instead of exclusive fullscreen.
			Some(winit::window::Fullscreen::Borderless(Some(use_monitor)))
		} else {
			// NOTE: This is specifically *exclusive* fullscreen, which gets ignored on Wayland.
			// Therefore, it might be a good idea to hide such an option in UI from the end user on Wayland.
			// TODO: Use VK_EXT_full_screen_exclusive to minimize latency (usually only available on Windows)
			if let Some(fullscreen_video_mode) = current_video_mode {
				Some(winit::window::Fullscreen::Exclusive(fullscreen_video_mode.clone()))
			} else {
				log::warn!("The current monitor's video mode could not be determined. Fullscreen mode is unavailable.");
				None
			}
		}
	} else {
		None
	}
}
