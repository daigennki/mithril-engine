/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use std::sync::Arc;
use std::time::Duration;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::{Image, ImageUsage};
use vulkano::swapchain::{
	ColorSpace, PresentMode, Surface, SurfaceInfo, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::{FenceSignalFuture, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::event_loop::EventLoop;
use winit::window::{Fullscreen, Window, WindowBuilder};

use crate::EngineError;

// Pairs of surface format and color space we can support.
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
	graphics_queue: Arc<Queue>,
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
	pub fn new(graphics_queue: Arc<Queue>, event_loop: &EventLoop<()>, window_title: &str) -> crate::Result<Self>
	{
		let vk_dev = graphics_queue.device().clone();
		let pd = vk_dev.physical_device();
		let window = create_window(event_loop, window_title)?;
		let surface = Surface::from_window(vk_dev.instance().clone(), window.clone())?;

		// Find the first intersection between the format candidates and the formats supported by
		// the physical device.
		let surface_formats = pd.surface_formats(&surface, SurfaceInfo::default())?;
		let (image_format, image_color_space) = FORMAT_CANDIDATES
			.into_iter()
			.find(|candidate| surface_formats.contains(candidate))
			.unwrap();

		let surface_present_modes: smallvec::SmallVec<[_; 4]> =
			pd.surface_present_modes(&surface, SurfaceInfo::default())?.collect();
		log::info!("Supported present modes: {:?}", &surface_present_modes);
		let present_mode_regex = regex::Regex::new("--present_mode=(?<value>\\w+)").unwrap();
		let present_mode = std::env::args()
			.collect::<Vec<_>>()
			.iter()
			.find_map(|arg| present_mode_regex.captures(arg))
			.and_then(|caps| caps.name("value"))
			.and_then(|value_match| match value_match.as_str() {
				"Immediate" => Some(PresentMode::Immediate),
				"Mailbox" => Some(PresentMode::Mailbox),
				"Fifo" => Some(PresentMode::Fifo),
				"FifoRelaxed" => Some(PresentMode::FifoRelaxed),
				_ => None,
			})
			.filter(|mode| {
				let mode_supported = surface_present_modes.contains(mode);
				if !mode_supported {
					log::warn!("Requested present mode `{mode:?}` is not supported, falling back to `Fifo`...");
				}
				mode_supported
			})
			.unwrap_or(PresentMode::Fifo);

		let surface_caps = pd.surface_capabilities(&surface, SurfaceInfo::default())?;
		let create_info = SwapchainCreateInfo {
			min_image_count: surface_caps.min_image_count.max(2),
			image_extent: window.inner_size().into(),
			image_format,
			image_color_space,
			image_usage: ImageUsage::TRANSFER_DST,
			present_mode,
			..Default::default()
		};
		let (swapchain, images) = vulkano::swapchain::Swapchain::new(vk_dev, surface, create_info)
			.map_err(|e| EngineError::new("failed to create swapchain", e.unwrap()))?;
		log::info!(
			"Created a swapchain with {} images ({:?}, {:?}, {:?})",
			images.len(),
			image_format,
			image_color_space,
			present_mode,
		);

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
			graphics_queue,
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
		self.window
			.set_fullscreen(fullscreen.then_some(winit::window::Fullscreen::Borderless(None)));
	}
	pub fn is_fullscreen(&self) -> bool
	{
		self.window.fullscreen().is_some()
	}

	pub fn is_minimized(&self) -> bool
	{
		self.window.is_minimized().unwrap_or(false)
	}

	pub fn get_next_image(&mut self) -> crate::Result<Option<Arc<Image>>>
	{
		if let Some(f) = self.submission_future.as_mut() {
			f.cleanup_finished();
		}

		// If the window is minimized, don't acquire an image. This must be done because the window
		// (often on Windows) may report an inner width or height of 0, which we can't resize the
		// swapchain to. Presenting swapchain images anyways would cause an "out of date" error too.
		if self.window.is_minimized().unwrap_or(false) {
			return Ok(None);
		}

		// Recreate the swapchain if the surface's properties changed (e.g. window size changed).
		let prev_extent = self.swapchain.image_extent();
		let new_inner_size = self.window.inner_size().into();
		self.extent_changed = prev_extent != new_inner_size;
		if self.extent_changed || self.recreate_pending {
			log::info!("Recreating swapchain; size will change from {prev_extent:?} to {new_inner_size:?}");

			// Set minimum size here to adapt to any DPI scale factor changes that may occur.
			self.window.set_min_inner_size(Some(winit::dpi::PhysicalSize::new(1280, 720)));

			let create_info = SwapchainCreateInfo {
				image_extent: new_inner_size,
				..self.swapchain.create_info()
			};
			let (new_swapchain, new_images) = self
				.swapchain
				.recreate(create_info)
				.map_err(|e| EngineError::new("failed to recreate swapchain", e.unwrap()))?;

			self.swapchain = new_swapchain;
			self.images = new_images;
			self.recreate_pending = false;
		}

		let acquire_result = vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None);
		let (image_num, suboptimal, acquire_future) = match acquire_result {
			Err(Validated::Error(VulkanError::OutOfDate)) => {
				// If the swapchain is out of date, don't return an image;
				// recreate the swapchain before the next image is acquired.
				log::info!("Swapchain is out of date, recreate pending...");
				self.recreate_pending = true;
				return Ok(None);
			}
			other => other?,
		};
		if suboptimal {
			log::info!("Swapchain is suboptimal, recreate pending...");
			self.recreate_pending = true;
		}

		// Panic if an image has been acquired without being submitted
		assert!(self.acquire_future.replace(acquire_future).is_none());

		Ok(Some(self.images[image_num as usize].clone()))
	}

	/// Submit a primary command buffer to the graphics queue, and then present the resulting image.
	/// If `after` is `Some`, the submitted work won't begin until after it.
	///
	/// If this is called with no image acquired, it'll submit the command buffer without presenting
	/// an image, which may be useful when the window is minimized.
	pub fn submit(
		&mut self,
		cb: Arc<PrimaryAutoCommandBuffer>,
		after: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,
	) -> crate::Result<()>
	{
		let mut joined_futures = vulkano::sync::future::now(self.graphics_queue.device().clone()).boxed_send_sync();

		if let Some(f) = after {
			// Ideally we'd use a semaphore instead of a fence here, but it's borked in Vulkano right now.
			f.wait(None)?;
			joined_futures = Box::new(joined_futures.join(f));
		}

		if let Some(f) = self.submission_future.take() {
			// Wait for the previous submission to finish to make sure resources are no longer in use.
			f.wait(Some(Duration::from_secs(5)))?;
			joined_futures = Box::new(joined_futures.join(f));
		}

		self.sleep_and_calculate_delta();

		self.submission_future = match self.acquire_future.take() {
			Some(acquire_future) => {
				let image_index = acquire_future.image_index();
				let present_info = SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index);

				let submit_result = joined_futures
					.join(acquire_future)
					.then_execute(self.graphics_queue.clone(), cb)
					.unwrap()
					.then_swapchain_present(self.graphics_queue.clone(), present_info)
					.boxed_send_sync()
					.then_signal_fence_and_flush();

				match submit_result {
					Err(Validated::Error(VulkanError::OutOfDate)) => {
						// If the swapchain is out of date, don't present;
						// recreate the swapchain before the next image is acquired.
						log::info!("Swapchain is out of date, recreate pending...");
						self.recreate_pending = true;
						None
					}
					other => Some(other?),
				}
			}
			None => joined_futures
				.then_execute(self.graphics_queue.clone(), cb)
				.unwrap()
				.boxed_send_sync()
				.then_signal_fence_and_flush()?
				.into(),
		};

		Ok(())
	}

	/// Sleep this thread so that the framerate stays below the limit, then calculate the delta time.
	///
	/// NOTE: High resolution timer on Windows is only available since Rust 1.75.
	fn sleep_and_calculate_delta(&mut self)
	{
		let sleep_dur = self.last_frame_presented + self.frame_time_min_limit - std::time::Instant::now();
		if sleep_dur > Duration::ZERO {
			std::thread::sleep(sleep_dur);
		}

		let now = std::time::Instant::now();
		self.frame_time = now - self.last_frame_presented;
		self.last_frame_presented = now;
	}

	pub fn graphics_queue_family_index(&self) -> u32
	{
		self.graphics_queue.queue_family_index()
	}

	pub fn image_count(&self) -> usize
	{
		self.images.len()
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
	pub fn delta(&self) -> Duration
	{
		self.frame_time
	}
}

fn create_window(event_loop: &EventLoop<()>, window_title: &str) -> crate::Result<Arc<Window>>
{
	let primary_monitor = event_loop.primary_monitor();

	// If "--fullscreen" was specified in the arguments, use "borderless" fullscreen on the primary
	// monitor. ("exclusive" fullscreen provides no benefits for Vulkan)
	let (fullscreen, inner_size) = std::env::args()
		.find(|arg| arg == "--fullscreen")
		.map(|_| {
			let inner_size = primary_monitor.as_ref().map_or_else(|| [1280, 720].into(), |mon| mon.size());
			(Some(Fullscreen::Borderless(primary_monitor.clone())), inner_size)
		})
		.unwrap_or_else(|| (None, [1280, 720].into()));

	let window = WindowBuilder::new()
		.with_min_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
		.with_inner_size(inner_size)
		.with_title(window_title)
		.with_fullscreen(fullscreen)
		.with_decorations(!std::env::args().any(|arg| arg == "--noborder"))
		.build(event_loop)
		.map_err(|e| EngineError::new("failed to create window", e))?;

	// Center the window on the primary monitor, if the primary monitor could be determined.
	//
	// On Wayland, this does nothing because `set_outer_position` is unsupported and
	// `primary_monitor` returns `None`. That's fine since Wayland seems to usually center it on the
	// current monitor by default.
	if let Some(some_mon) = primary_monitor {
		let mon_pos: [i32; 2] = some_mon.position().into();
		let mon_size: [u32; 2] = some_mon.size().into();
		let mon_size_half: IVec2 = (UVec2::from(mon_size) / 2).try_into().unwrap();
		let mon_center = IVec2::from(mon_pos) + mon_size_half;
		let outer_size: [u32; 2] = window.outer_size().into();
		let outer_size_half: IVec2 = (UVec2::from(outer_size) / 2).try_into().unwrap();
		let outer_pos = winit::dpi::Position::Physical((mon_center - outer_size_half).to_array().into());
		window.set_outer_position(outer_pos);
	}

	Ok(Arc::new(window))
}
