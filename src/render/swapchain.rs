/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;
use std::time::Duration;
use vulkano::command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{view::ImageView, ImageUsage};
use vulkano::swapchain::{
	ColorSpace, PresentMode, Surface, SurfaceInfo, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::{FenceSignalFuture, GpuFuture, NowFuture};
use vulkano::{Validated, VulkanError};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use crate::GenericEngineError;

pub struct Swapchain
{
	window: Arc<Window>,
	swapchain: Arc<vulkano::swapchain::Swapchain>,
	image_views: Vec<Arc<ImageView>>,

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
	pub fn new(vk_dev: Arc<Device>, event_loop: &EventLoop<()>, window_title: &str) -> Result<Self, GenericEngineError>
	{
		let window = create_window(event_loop, window_title)?;
		let surface = Surface::from_window(vk_dev.instance().clone(), window.clone())?;

		let pd = vk_dev.physical_device();
		let surface_caps = pd.surface_capabilities(&surface, SurfaceInfo::default())?;
		let surface_formats = pd.surface_formats(&surface, SurfaceInfo::default())?;
		let surface_present_modes = pd.surface_present_modes(&surface, SurfaceInfo::default())?;

		log::info!("Available surface format and color space combinations:");
		surface_formats.iter().for_each(|f| log::info!("{:?}", f));

		log::info!("Available surface present modes: {:?}", Vec::from_iter(surface_present_modes));

		// Pairs of format and color space we can support.
		let mut format_candidates = vec![
			// HDR via extended sRGB linear image
			// (disabled for now since this is sometimes "supported" on Windows when HDR is disabled for some reason)
			//(Format::R16G16B16A16_SFLOAT, ColorSpace::ExtendedSrgbLinear),

			// sRGB 10bpc
			(Format::A2B10G10R10_UNORM_PACK32, ColorSpace::SrgbNonLinear),
			// sRGB 8bpc
			(Format::B8G8R8A8_UNORM, ColorSpace::SrgbNonLinear),
		];

		// Find the intersection between the format candidates and the formats supported by the physical device,
		// then get the first one remaining.
		format_candidates.retain(|candidate| surface_formats.contains(candidate));
		let (image_format, image_color_space) = format_candidates[0];

		let image_usage = (image_color_space == ColorSpace::SrgbNonLinear)
			.then_some(ImageUsage::COLOR_ATTACHMENT)
			.unwrap_or(ImageUsage::TRANSFER_DST);

		let create_info = SwapchainCreateInfo {
			min_image_count: surface_caps.min_image_count.max(2),
			image_extent: window.inner_size().into(),
			image_format,
			image_color_space,
			image_usage,
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

		let mut image_views = Vec::with_capacity(images.len());
		for img in images {
			image_views.push(ImageView::new_default(img)?);
		}

		// TODO: load this from config
		let fps_max = 360;

		// Linux and Windows have different sleep overshoot, so different values are used for each.
		#[cfg(target_family = "windows")]
		const SLEEP_OVERSHOOT: Duration = Duration::from_micros(260);
		#[cfg(not(target_family = "windows"))]
		const SLEEP_OVERSHOOT: Duration = Duration::from_micros(50);

		// Subtract to account for sleep overshoot.
		let frame_time_min_limit = (Duration::from_secs(1) / fps_max)
			.checked_sub(SLEEP_OVERSHOOT)
			.unwrap_or_default();

		Ok(Swapchain {
			window,
			swapchain,
			image_views,
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
		self.window
			.set_fullscreen(fullscreen.then_some(winit::window::Fullscreen::Borderless(None)));
	}
	pub fn is_fullscreen(&self) -> bool
	{
		self.window.fullscreen().is_some()
	}

	/// Get the next swapchain image.
	pub fn get_next_image(&mut self) -> Result<Arc<ImageView>, GenericEngineError>
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
			log::info!(
				"Recreating swapchain; size will change from {:?} to {:?}",
				prev_extent,
				new_inner_size
			);

			// set minimum size here to make sure we adapt to any DPI scale factor changes that may arise
			self.window.set_min_inner_size(Some(winit::dpi::PhysicalSize::new(1280, 720)));

			let create_info = SwapchainCreateInfo {
				image_extent: new_inner_size,
				..self.swapchain.create_info()
			};
			let (new_swapchain, new_images) = self.swapchain.recreate(create_info)?;

			let mut new_image_views = Vec::with_capacity(new_images.len());
			for img in new_images {
				new_image_views.push(ImageView::new_default(img)?);
			}

			self.swapchain = new_swapchain;
			self.image_views = new_image_views;
			self.recreate_pending = false;
		}

		let timeout = Some(Duration::from_secs(5));
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

		Ok(self.image_views[image_num as usize].clone())
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

		if let Some(f) = self.submission_future.take() {
			// wait for the previous submission to finish, to make sure resources are no longer in use
			match f.wait(Some(std::time::Duration::from_secs(5))) {
				Ok(()) => (),
				Err(Validated::Error(VulkanError::Timeout)) => return Err("Graphics submission took too long!".into()),
				Err(e) => return Err(Box::new(e)),
			}
			joined_futures = Box::new(joined_futures.join(f));
		}

		if let Some(f) = after {
			// Ideally we'd use a semaphore instead of a fence here, but apprently it's borked in Vulkano right now.
			match f.wait(Some(Duration::from_secs(5))) {
				Ok(()) => (),
				Err(Validated::Error(VulkanError::Timeout)) => return Err("Transfer submission took too long!".into()),
				Err(e) => return Err(Box::new(e)),
			}
			joined_futures = Box::new(joined_futures.join(f));
		}

		self.sleep_and_calculate_delta();

		let submission_future = match self.acquire_future.take() {
			Some(acquire_future) => {
				let present_info =
					SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), acquire_future.image_index());

				joined_futures
					.join(acquire_future)
					.then_execute(queue.clone(), cb)?
					.then_swapchain_present(queue, present_info)
					.boxed_send_sync()
					.then_signal_fence_and_flush()?
			}
			None => joined_futures
				.then_execute(queue.clone(), cb)?
				.boxed_send_sync()
				.then_signal_fence_and_flush()?,
		};
		self.submission_future = Some(submission_future);

		Ok(())
	}

	/// Sleep this thread so that the framerate stays below the limit, then calculate the delta time.
	/// NOTE: High resolution timer on Windows is only available since Rust 1.75 beta.
	fn sleep_and_calculate_delta(&mut self)
	{
		let dur = self.last_frame_presented + self.frame_time_min_limit - std::time::Instant::now();
		if dur > Duration::ZERO {
			std::thread::sleep(dur);
		}

		let now = std::time::Instant::now();
		let dur = now - self.last_frame_presented;
		self.last_frame_presented = now;
		self.frame_time = dur;
	}

	pub fn image_count(&self) -> usize
	{
		self.image_views.len()
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

fn create_window(event_loop: &EventLoop<()>, window_title: &str) -> Result<Arc<Window>, GenericEngineError>
{
	let use_monitor = event_loop
		.primary_monitor()
		.or_else(|| event_loop.available_monitors().next())
		.ok_or("No monitors are available!")?;

	// If "-fullscreen" was specified in the arguments, use the current video mode with winit's "borderless" fullscreen.
	// winit also offers an "exclusive" fullscreen option, but for Vulkan, it provides no benefits.
	// TODO: load this from config
	let fullscreen_mode = std::env::args()
		.find(|arg| arg == "-fullscreen")
		.map(|_| winit::window::Fullscreen::Borderless(Some(use_monitor.clone())));
	if fullscreen_mode.is_some() {
		let mon_name = use_monitor.name().unwrap_or_else(|| "[no longer exists]".to_string());
		let mon_size = use_monitor.size();
		let refresh_rate = use_monitor.refresh_rate_millihertz().unwrap_or(0);
		log::info!(
			"Using fullscreen mode on monitor '{}' ({} x {} @ {}.{:03} Hz)",
			mon_name,
			mon_size.width,
			mon_size.height,
			refresh_rate / 1000,
			refresh_rate % 1000,
		);
	}

	// TODO: load window size from config
	let window_size = fullscreen_mode
		.as_ref()
		.map(|_| use_monitor.size())
		.unwrap_or_else(|| winit::dpi::PhysicalSize::new(1280, 720));

	// create window
	let window = WindowBuilder::new()
		.with_inner_size(window_size)
		.with_title(window_title)
		.with_fullscreen(fullscreen_mode)
		.build(&event_loop)?;

	Ok(Arc::new(window))
}
