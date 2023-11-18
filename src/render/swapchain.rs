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
	ColorSpace, PresentMode,
	Surface, SurfaceInfo, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::{
	future::{FenceSignalFuture, GpuFuture, NowFuture}
};
use vulkano::{Validated, VulkanError};
use winit::window::{Window, WindowBuilder};
use winit::monitor::{MonitorHandle, VideoMode};

use crate::GenericEngineError;

pub struct Swapchain
{
	window: Arc<Window>,
	swapchain: Arc<vulkano::swapchain::Swapchain>,
	images: Vec<Arc<Image>>,

	extent_changed: bool, // `true` if image extent changed since the last presentation
	recreate_pending: bool,

	acquire_future: Option<SwapchainAcquireFuture>,
	submission_future: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,

	last_frame_presented: std::time::Instant,
	frame_time: std::time::Duration,
	frame_time_min_limit: std::time::Duration, // minimum frame time, used for framerate limit
}
impl Swapchain
{
	pub fn new(
		vk_dev: Arc<Device>, 
		event_loop: &winit::event_loop::EventLoop<()>,
		window_title: &str
	) -> Result<Self, GenericEngineError>
	{
		let pd = vk_dev.physical_device();

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
		let surface_caps = pd.surface_capabilities(&surface, SurfaceInfo::default())?;
		let surface_formats = pd.surface_formats(&surface, SurfaceInfo::default())?;
		let surface_present_modes = pd.surface_present_modes(&surface, SurfaceInfo::default())?;

		log::info!("Available surface format and color space combinations:");
		surface_formats.iter().for_each(|f| log::info!("{:?}", f));

		log::info!("Available surface present modes: {:?}", Vec::from_iter(surface_present_modes));

		// Pairs of format and color space we can support
		let mut format_candidates = vec![
			// HDR via extended sRGB linear image
			// (disabled for now since this is sometimes "supported" on Windows when HDR is disabled for some reason)
			//(Format::R16G16B16A16_SFLOAT, ColorSpace::ExtendedSrgbLinear),

			// sRGB image automatically converts from linear to non-linear
			(Format::B8G8R8A8_SRGB, ColorSpace::SrgbNonLinear),

			// Requires separate conversion from linear to non-linear, but is supported on practically any GPU,
			// so at least this pair must be retained. See `RenderTarget::copy_to_swapchain` for the conversion.
			(Format::B8G8R8A8_UNORM, ColorSpace::SrgbNonLinear),
		];

		// Find the intersection between the format candidates and the formats supported by the physical device,
		// then get the first one remaining.
		format_candidates.retain(|candidate| surface_formats.contains(candidate));
		let (image_format, image_color_space) = format_candidates[0];

		let create_info = SwapchainCreateInfo {
			min_image_count: surface_caps.min_image_count.max(2),
			image_extent: window_size.into(),
			image_format,
			image_color_space,
			image_usage: ImageUsage::TRANSFER_DST,
			present_mode: PresentMode::Fifo,
			..Default::default()
		};
		let (swapchain, images) = vulkano::swapchain::Swapchain::new(vk_dev.clone(), surface, create_info)?;
		log::info!(
			"Created a swapchain with {} images (format {:?}, color space {:?})",
			images.len(),
			image_format,
			image_color_space
		);

		// TODO: load this from config
		let fps_max = 360;
		let frame_time_min_limit = std::time::Duration::from_secs(1) / fps_max;

		Ok(Swapchain {
			window: window_arc,
			swapchain,
			images,
			extent_changed: false,
			recreate_pending: false,
			acquire_future: None,
			submission_future: None,
			last_frame_presented: std::time::Instant::now(),
			frame_time: std::time::Duration::ZERO,
			frame_time_min_limit,
		})
	}

	pub fn set_fullscreen(&self, fullscreen: bool)
	{
		self.window.set_fullscreen(fullscreen.then_some(winit::window::Fullscreen::Borderless(None)));
	}
	pub fn is_fullscreen(&self) -> bool
	{
		self.window.fullscreen().is_some()
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
		let prev_extent = self.swapchain.image_extent();
		let new_inner_size = self.window.inner_size().into();
		self.extent_changed = prev_extent != new_inner_size;
		if self.extent_changed || self.recreate_pending {
			log::info!("Recreating swapchain; size will change from {:?} to {:?}", prev_extent, new_inner_size);

			// set minimum size here to make sure we adapt to any DPI scale factor changes that may arise
			self.window.set_min_inner_size(Some(winit::dpi::PhysicalSize::new(1280, 720)));

			let create_info = SwapchainCreateInfo {
				image_extent: new_inner_size,
				..self.swapchain.create_info()
			};
			let (new_swapchain, new_images) = self.swapchain.recreate(create_info)?;
			self.swapchain = new_swapchain;
			self.images = new_images;
			self.recreate_pending = false;
		}

		let timeout = Some(std::time::Duration::from_secs(5));
		let acquire_result = vulkano::swapchain::acquire_next_image(self.swapchain.clone(), timeout);
		let (image_num, suboptimal, acquire_future) = match acquire_result {
			Ok(ok) => ok,
			Err(Validated::Error(VulkanError::Timeout)) => {
				return Err("Swapchain image took too long to become available!".into())
			}
			Err(e) => return Err(Box::new(e)),
		};
		if suboptimal {
			log::warn!("Swapchain is suboptimal! Recreate pending...");
			self.recreate_pending = true;
		}
		self.acquire_future = Some(acquire_future);

		Ok(self.images[image_num as usize].clone())
	}

	/// Submit a primary command buffer's commands (where the command buffer is expected to manipulate the currently acquired
	/// swapchain image, usually blitting to it) and then present the resulting image.
	/// Optionally, a future `after` to wait for (usually for joining submitted transfers on another queue) can be given, so
	/// that graphics operations don't begin until after that future is reached.
	///
	/// If this is called without acquiring an image, it'll assume that you want to submit the contents of the command buffer
	/// to the graphics queue without presenting an image, which may be useful when the window is minimized.
	pub fn submit(
		&mut self,
		cb: Arc<PrimaryAutoCommandBuffer>,
		queue: Arc<Queue>,
		after: Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>,
	) -> Result<(), GenericEngineError>
	{
		let mut joined_futures = vulkano::sync::future::now(queue.device().clone()).boxed_send_sync();

		// To keep frame presentation timing stable, we must sleep before waiting for the fence.
		self.sleep_and_calculate_delta();

		if let Some(f) = self.submission_future.take() {
			// wait for the previous submission to finish, to make sure resources are no longer in use
			match f.wait(Some(std::time::Duration::from_secs(5))) {
				Ok(()) => (),
				Err(Validated::Error(VulkanError::Timeout)) => {
					return Err("Graphics submission took too long!".into())
				}
				Err(e) => return Err(Box::new(e)),
			}
			joined_futures = Box::new(joined_futures.join(f));
		}

		if let Some(f) = after {
			// Ideally we'd use a semaphore instead of a fence here, but apprently it's borked in Vulkano right now.
			match f.wait(Some(std::time::Duration::from_secs(5))) {
				Ok(()) => (),
				Err(Validated::Error(VulkanError::Timeout)) => {
					return Err("Transfer submission took too long!".into())
				}
				Err(e) => return Err(Box::new(e)),
			}
			joined_futures = Box::new(joined_futures.join(f));
		}

		let submission_future = match self.acquire_future.take() {
			Some(acquire_future) => {
				let present_info = SwapchainPresentInfo::swapchain_image_index(
					self.swapchain.clone(),
					acquire_future.image_index()
				);

				joined_futures
					.join(acquire_future)
					.then_execute(queue.clone(), cb)?
					.then_swapchain_present(queue, present_info)
					.boxed_send_sync()
					.then_signal_fence_and_flush()?
			}
			None => {
				joined_futures
					.then_execute(queue.clone(), cb)?
					.boxed_send_sync()
					.then_signal_fence_and_flush()?
			}
		};
		self.submission_future = Some(submission_future);

		Ok(())
	}

	/// Sleep this thread so that the framerate stays below the limit, then calculate the delta time.
	fn sleep_and_calculate_delta(&mut self)
	{
		let sleep_until = self.last_frame_presented + self.frame_time_min_limit;
		if sleep_until > std::time::Instant::now() {
			// subtract to account for sleep overshoot
			let sleep_dur = (sleep_until - std::time::Instant::now())
				.checked_sub(std::time::Duration::from_micros(50));

			if let Some(d) = sleep_dur {
				std::thread::sleep(d);
			}
		}

		let now = std::time::Instant::now();
		let dur = now - self.last_frame_presented;
		self.last_frame_presented = now;
		self.frame_time = dur;
	}

	pub fn image_count(&self) -> usize
	{
		self.images.len()
	}

	pub fn format(&self) -> Format
	{
		self.swapchain.image_format()
	}
	pub fn color_space(&self) -> ColorSpace
	{
		self.swapchain.image_color_space()
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
	if current_video_mode.is_none() {
		log::warn!("The current monitor's video mode could not be determined. Fullscreen mode is unavailable.");
	}

	let mut video_modes: Vec<_> = mon.video_modes().collect();
	log::info!("All video modes supported by current monitor (\"{mon_name}\"):");

	// filter the video modes to those with >=1280 width, >=720 height, and not vertical (width <= height)
	video_modes.retain(|video_mode| {
		// print unfiltered video modes while we're at it
		let video_mode_suffix = current_video_mode
			.as_ref()
			.filter(|cur_vm| video_mode == *cur_vm)
			.map(|_| " <- likely current primary monitor video mode")
			.unwrap_or_default();

		log::info!("{}{}", video_mode, video_mode_suffix);

		let size = video_mode.size();
		size.width >= 1280 && size.height >= 720 && size.width >= size.height
	});

	// filter the video modes to the highest refresh rate for each size
	// (sort beforehand so that highest refresh rate for each size comes first,
	// then remove duplicates with the same size and bit depth)
	video_modes.sort();
	video_modes.dedup_by_key(|video_mode| (video_mode.size(), video_mode.bit_depth()));

	log::info!("Filtered video modes:");
	video_modes.iter().for_each(|vm| log::info!("{vm}"));

	// Determine the appropriate fullscreen mode based on the arguments given to the executable.
	//
	// If "-fullscreen" and "-borderless" were both specified, make a borderless window filling the entire monitor
	// instead of exclusive fullscreen.
	// If only "-fullscreen" was specified, use the current video mode with exclusive fullscreen.
	//
	// NOTE: Exclusive fullscreen gets ignored on Wayland.
	// Therefore, it might be a good idea to hide such an option in UI from the end user on Wayland.
	//
	// TODO: load this from config
	let fullscreen_mode = std::env::args()
		.find(|arg| arg == "-fullscreen")
		.and_then(|_| {
			std::env::args()
				.find(|arg| arg == "-borderless")
				.map(|_| winit::window::Fullscreen::Borderless(Some(mon)))
				.or_else(|| current_video_mode.clone().map(|vm| winit::window::Fullscreen::Exclusive(vm)))
		});

	(current_video_mode, fullscreen_mode)
}

