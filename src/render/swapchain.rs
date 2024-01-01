/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
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

use crate::EngineError;

// Pairs of format and color space we can support.
const FORMAT_CANDIDATES: [(Format, ColorSpace); 2] = [
	// HDR via extended sRGB linear image
	// (disabled for now since this is sometimes "supported" on Windows when HDR is disabled for some reason)
	//(Format::R16G16B16A16_SFLOAT, ColorSpace::ExtendedSrgbLinear),

	// sRGB 10bpc
	(Format::A2B10G10R10_UNORM_PACK32, ColorSpace::SrgbNonLinear),
	// sRGB 8bpc
	(Format::B8G8R8A8_UNORM, ColorSpace::SrgbNonLinear),
];

// Windows and Linux have different sleep overshoot, so different values are used for each.
#[cfg(target_family = "windows")]
const SLEEP_OVERSHOOT: Duration = Duration::from_micros(260);
#[cfg(not(target_family = "windows"))]
const SLEEP_OVERSHOOT: Duration = Duration::from_micros(50);

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
	pub fn new(vk_dev: Arc<Device>, event_loop: &EventLoop<()>, window_title: &str) -> crate::Result<Self>
	{
		let window = create_window(event_loop, window_title)?;
		let surface = Surface::from_window(vk_dev.instance().clone(), window.clone())?;

		let pd = vk_dev.physical_device();
		let surface_caps = pd.surface_capabilities(&surface, SurfaceInfo::default())?;
		let surface_formats = pd.surface_formats(&surface, SurfaceInfo::default())?;
		let surface_present_modes = pd.surface_present_modes(&surface, SurfaceInfo::default())?;

		log::info!("Available surface format and color space combinations:");
		surface_formats.iter().for_each(|f| log::info!("- {f:?}"));

		log::info!("Available surface present modes: {:?}", Vec::from_iter(surface_present_modes));

		// Find the intersection between the format candidates and the formats supported by the physical device,
		// then get the first one remaining.
		let (image_format, image_color_space) = FORMAT_CANDIDATES
			.into_iter()
			.filter(|candidate| surface_formats.contains(candidate))
			.next()
			.unwrap();

		let image_usage = (image_color_space == ColorSpace::SrgbNonLinear)
			.then_some(ImageUsage::STORAGE)
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
		let (swapchain, images) = vulkano::swapchain::Swapchain::new(vk_dev.clone(), surface, create_info)
			.map_err(|e| EngineError::new("failed to create swapchain", e.unwrap()))?;
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

		// Set the framerate limit
		let fps_max_regex = regex::Regex::new("--fps_max=(?<value>\\d+)").unwrap();
		let fps_max = std::env::args()
			.collect::<Vec<_>>()
			.iter()
			.find_map(|arg| fps_max_regex.captures(arg))
			.and_then(|caps| caps.name("value"))
			.and_then(|value_match| value_match.as_str().parse().ok())
			.unwrap_or(360);

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
	pub fn get_next_image(&mut self) -> crate::Result<Option<(u32, Arc<ImageView>)>>
	{
		// Panic if this function is called when an image has already been acquired without being submitted
		assert!(self.acquire_future.is_none());

		// clean up resources from finished submissions
		if let Some(f) = self.submission_future.as_mut() {
			f.cleanup_finished();
		}

		// If the window is minimized, don't acquire an image.
		// We have to do this because sometimes (often on Windows) the window may report an inner width or height of 0,
		// which we can't resize the swapchain to. We can't keep presenting swapchain images without causing an "out of date"
		// error either, so we just have to not present any images.
		if self.window.is_minimized().unwrap_or(false) {
			return Ok(None);
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
			let (new_swapchain, new_images) = self
				.swapchain
				.recreate(create_info)
				.map_err(|e| EngineError::new("failed to recreate swapchain", e.unwrap()))?;

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
			Err(Validated::Error(VulkanError::OutOfDate)) => {
				// If the swapchain is out of date, don't return an image;
				// recreate the swapchain before the next image is acquired.
				log::warn!("Swapchain is out of date! Recreate pending...");
				self.recreate_pending = true;
				return Ok(None);
			}
			Err(Validated::Error(VulkanError::Timeout)) => {
				return Err("Swapchain image took too long to become available!".into())
			}
			Err(e) => return Err(e.into()),
		};
		if suboptimal {
			log::warn!("Swapchain is suboptimal! Recreate pending...");
			self.recreate_pending = true;
		}
		self.acquire_future = Some(acquire_future);

		Ok(Some((image_num, self.image_views[image_num as usize].clone())))
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
	) -> crate::Result<()>
	{
		let mut joined_futures = vulkano::sync::future::now(queue.device().clone()).boxed_send_sync();

		if let Some(f) = self.submission_future.take() {
			// wait for the previous submission to finish, to make sure resources are no longer in use
			match f.wait(Some(std::time::Duration::from_secs(5))) {
				Ok(()) => (),
				Err(Validated::Error(VulkanError::Timeout)) => return Err("Graphics submission took too long!".into()),
				Err(e) => return Err(e.into()),
			}
			joined_futures = Box::new(joined_futures.join(f));
		}

		if let Some(f) = after {
			// Ideally we'd use a semaphore instead of a fence here, but apprently it's borked in Vulkano right now.
			match f.wait(Some(Duration::from_secs(5))) {
				Ok(()) => (),
				Err(Validated::Error(VulkanError::Timeout)) => return Err("Transfer submission took too long!".into()),
				Err(e) => return Err(e.into()),
			}
			joined_futures = Box::new(joined_futures.join(f));
		}

		self.sleep_and_calculate_delta();

		let submission_future = match self.acquire_future.take() {
			Some(acquire_future) => {
				let present_info =
					SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), acquire_future.image_index());

				let submit_result = joined_futures
					.join(acquire_future)
					.then_execute(queue.clone(), cb)
					.unwrap()
					.then_swapchain_present(queue, present_info)
					.boxed_send_sync()
					.then_signal_fence_and_flush();

				match submit_result {
					Ok(f) => Some(f),
					Err(Validated::Error(VulkanError::OutOfDate)) => {
						// If the swapchain is out of date, don't present;
						// recreate the swapchain before the next image is acquired.
						log::warn!("Swapchain is out of date! Recreate pending...");
						self.recreate_pending = true;
						None
					}
					Err(e) => return Err(e.into()),
				}
			}
			None => {
				let f = joined_futures
					.then_execute(queue.clone(), cb)
					.unwrap()
					.boxed_send_sync()
					.then_signal_fence_and_flush()?;

				Some(f)
			}
		};
		self.submission_future = submission_future;

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

	pub fn get_images(&self) -> &Vec<Arc<ImageView>>
	{
		&self.image_views
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

	/// Get the delta time for last frame.
	pub fn delta(&self) -> std::time::Duration
	{
		self.frame_time
	}
}

fn create_window(event_loop: &EventLoop<()>, window_title: &str) -> crate::Result<Arc<Window>>
{
	let mon = event_loop
		.primary_monitor()
		.or_else(|| event_loop.available_monitors().next())
		.ok_or("No monitors are available!")?;

	let mon_name = mon.name().unwrap_or_else(|| "[no longer exists]".to_string());
	let mon_size: [u32; 2] = mon.size().into();
	let refresh_rate = mon.refresh_rate_millihertz().unwrap_or(0);
	log::info!(
		"Guessed primary monitor: '{}' ({} x {} @ {}.{:03} Hz)",
		mon_name,
		mon_size[0],
		mon_size[1],
		refresh_rate / 1000,
		refresh_rate % 1000,
	);

	// If "--fullscreen" was specified in the arguments, use winit's "borderless" fullscreen on the primary monitor.
	// winit also offers an "exclusive" fullscreen option, but for Vulkan, it provides no benefits.
	let (fullscreen, inner_size) = std::env::args()
		.find(|arg| arg == "--fullscreen")
		.map(|_| (Some(winit::window::Fullscreen::Borderless(Some(mon.clone()))), mon.size()))
		.unwrap_or_else(|| (None, [1280, 720].into()));

	let window = WindowBuilder::new()
		.with_inner_size(inner_size)
		.with_title(window_title)
		.with_fullscreen(fullscreen)
		.with_decorations(std::env::args().find(|arg| arg == "--noborder").is_none())
		.build(&event_loop)
		.map_err(|e| EngineError::new("failed to create window", e))?;

	// Center the window on the primary monitor.
	//
	// winit says that `set_outer_position` is unsupported on Wayland,
	// but that shouldn't be a problem since Wayland already centers the window by default
	// (albeit on the "current" monitor rather than the "primary" monitor).
	let mon_pos: [i32; 2] = mon.position().into();
	let mon_size_half: IVec2 = (UVec2::from(mon_size) / 2).try_into().unwrap();
	let mon_center = IVec2::from(mon_pos) + mon_size_half;
	let outer_size: [u32; 2] = window.outer_size().into();
	let outer_size_half: IVec2 = (UVec2::from(outer_size) / 2).try_into().unwrap();
	let outer_pos = winit::dpi::Position::Physical((mon_center - outer_size_half).to_array().into());
	window.set_outer_position(outer_pos);

	Ok(Arc::new(window))
}
