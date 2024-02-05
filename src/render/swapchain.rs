/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use std::sync::Arc;
use std::time::Duration;
use vulkano::command_buffer::PrimaryCommandBufferAbstract;
use vulkano::device::{physical::PhysicalDevice, Device, Queue};
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

const WINDOW_MIN_INNER_SIZE: winit::dpi::PhysicalSize<u32> = winit::dpi::PhysicalSize::new(1280, 720);

// Pairs of surface format and color space we can support.
const FORMAT_CANDIDATES: [(Format, ColorSpace); 2] = [
	// HDR via extended sRGB linear image
	// (disabled for now since this is sometimes "supported" on Windows when HDR is disabled for some reason)
	//(Format::R16G16B16A16_SFLOAT, ColorSpace::ExtendedSrgbLinear),

	// sRGB 10bpc
	(Format::A2B10G10R10_UNORM_PACK32, ColorSpace::SrgbNonLinear),
	// sRGB 8bpc (guaranteed to be supported by any physical device and surface)
	(Format::B8G8R8A8_UNORM, ColorSpace::SrgbNonLinear),
];

// The sleep overshoot that gets subtracted from the minimum frame time for the framerate limiter.
// Overshoot differs between Windows and Linux.
#[cfg(target_family = "windows")]
const SLEEP_OVERSHOOT: Duration = Duration::from_micros(260);
#[cfg(not(target_family = "windows"))]
const SLEEP_OVERSHOOT: Duration = Duration::from_micros(50);

pub struct Swapchain
{
	window: Arc<Window>,
	swapchain: Arc<vulkano::swapchain::Swapchain>,
	images: Vec<Arc<Image>>,
	submission_future: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,

	extent_changed: bool, // `true` if image extent changed since the last presentation
	recreate_pending: bool,
	last_frame_presented: std::time::Instant,
	frame_time: Duration,
	frame_time_min_limit: Duration, // minimum frame time, used for framerate limit
}
impl Swapchain
{
	pub fn new(device: Arc<Device>, event_loop: &EventLoop<()>, window_title: &str) -> crate::Result<Self>
	{
		let pd = device.physical_device();
		let window = create_window(event_loop, window_title)?;
		let surface = Surface::from_window(device.instance().clone(), window.clone())?;

		// Use the first format candidate supported by the physical device.
		let surface_formats = pd.surface_formats(&surface, SurfaceInfo::default())?;
		let (image_format, image_color_space) = FORMAT_CANDIDATES
			.into_iter()
			.find(|candidate| surface_formats.contains(candidate))
			.unwrap();

		let present_mode = get_configured_present_mode(pd, &surface)?;
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
		let (swapchain, images) = vulkano::swapchain::Swapchain::new(device, surface, create_info)?;

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
			.unwrap_or(400);
		let frame_time_min_limit = (Duration::from_secs(1) / fps_max)
			.checked_sub(SLEEP_OVERSHOOT)
			.unwrap_or_default();

		Ok(Swapchain {
			window,
			swapchain,
			images,
			submission_future: None,
			extent_changed: false,
			recreate_pending: false,
			last_frame_presented: std::time::Instant::now(),
			frame_time: std::time::Duration::ZERO,
			frame_time_min_limit,
		})
	}

	/// Acquire the next swapchain image. Returns `None` if the swapchain is out of date (it'll be
	/// recreated the next time this is called).
	pub fn get_next_image(&mut self) -> crate::Result<Option<(Arc<Image>, SwapchainAcquireFuture)>>
	{
		if let Some(f) = &mut self.submission_future {
			f.cleanup_finished();
		}

		// Recreate the swapchain if the surface's properties changed (e.g. window size changed),
		// but only if the new extent is valid (neither width nor height are 0).
		let prev_extent = self.swapchain.image_extent();
		let new_extent: [u32; 2] = self.window.inner_size().into();
		self.extent_changed = prev_extent != new_extent;
		if (self.extent_changed || self.recreate_pending) && new_extent[0] > 0 && new_extent[1] > 0 {
			log::info!("Recreating swapchain (size {prev_extent:?} -> {new_extent:?})");
			let create_info = SwapchainCreateInfo {
				image_extent: new_extent,
				..self.swapchain.create_info()
			};
			(self.swapchain, self.images) = self.swapchain.recreate(create_info)?;
		}

		let acquire_result = vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None);
		let (image_index, suboptimal, acquire_future) = match acquire_result {
			Err(Validated::Error(VulkanError::OutOfDate)) => {
				self.recreate_pending = true;
				return Ok(None);
			}
			other => other?,
		};
		self.recreate_pending = suboptimal;

		Ok(Some((self.images[image_index as usize].clone(), acquire_future)))
	}

	/// Submit a primary command buffer to the graphics queue, and then present the resulting image.
	///
	/// If the swapchain is out of date, this won't present an image, and the swapchain will be
	/// recreated the next time `get_next_image` is called.
	pub fn present(
		&mut self,
		queue: Arc<Queue>,
		cb: Arc<impl PrimaryCommandBufferAbstract + 'static>,
		acquire_future: SwapchainAcquireFuture,
		join_with: Option<impl GpuFuture + Send + Sync + 'static>,
	) -> crate::Result<()>
	{
		let image_index = acquire_future.image_index();
		let mut joined_futures = match self.submission_future.take() {
			Some(f) => {
				// Wait for the previous submission to finish to make sure resources are no longer in use.
				f.wait(Some(Duration::from_secs(5)))?;
				f.join(acquire_future).boxed_send_sync()
			}
			None => acquire_future.boxed_send_sync(),
		};
		if let Some(f) = join_with {
			joined_futures = Box::new(joined_futures.join(f));
		}

		// Sleep this thread to limit the framerate, then calculate the frame time.
		let sleep_dur = self.last_frame_presented + self.frame_time_min_limit - std::time::Instant::now();
		if sleep_dur > Duration::ZERO {
			std::thread::sleep(sleep_dur);
		}
		let now = std::time::Instant::now();
		self.frame_time = now - self.last_frame_presented;
		self.last_frame_presented = now;

		let present_info = SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index);
		let submit_result = joined_futures
			.then_execute(queue.clone(), cb)
			.unwrap()
			.then_swapchain_present(queue, present_info)
			.boxed_send_sync()
			.then_signal_fence_and_flush();

		self.submission_future = match submit_result {
			Err(Validated::Error(VulkanError::OutOfDate)) => {
				self.recreate_pending = true;
				None
			}
			other => Some(other?),
		};

		Ok(())
	}

	pub fn set_fullscreen(&self, fullscreen: bool)
	{
		self.window.set_fullscreen(fullscreen.then_some(Fullscreen::Borderless(None)));
	}
	pub fn is_fullscreen(&self) -> bool
	{
		self.window.fullscreen().is_some()
	}

	/// Set minimum size again to adapt to any DPI scale factor changes that may occur.
	pub fn reset_min_inner_size(&self)
	{
		self.window.set_min_inner_size(Some(WINDOW_MIN_INNER_SIZE));
	}

	pub fn is_minimized(&self) -> bool
	{
		self.window.is_minimized().unwrap_or(false)
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

	/// Get the frame time for last frame.
	pub fn delta(&self) -> Duration
	{
		self.frame_time
	}
}

fn create_window(event_loop: &EventLoop<()>, window_title: &str) -> crate::Result<Arc<Window>>
{
	let primary_monitor = event_loop.primary_monitor();

	// Use "borderless" fullscreen if requested. ("exclusive" fullscreen is meaningless for Vulkan)
	let fullscreen = std::env::args().any(|arg| arg == "--fullscreen");
	let inner_size = primary_monitor
		.as_ref()
		.filter(|_| fullscreen)
		.map_or(WINDOW_MIN_INNER_SIZE, |mon| mon.size());

	let window = WindowBuilder::new()
		.with_min_inner_size(WINDOW_MIN_INNER_SIZE)
		.with_inner_size(inner_size)
		.with_title(window_title)
		.with_fullscreen(fullscreen.then(|| Fullscreen::Borderless(primary_monitor.clone())))
		.with_decorations(!std::env::args().any(|arg| arg == "--noborder"))
		.build(event_loop)
		.map_err(|e| EngineError::new("failed to create window", e))?;

	// Center the window on the primary monitor, if the primary monitor could be determined.
	//
	// On Wayland, `set_outer_position` is unsupported and `primary_monitor` returns `None`, so this
	// does nothing. We'll just have to hope that the WM centers the window (KWin is one that does).
	if let Some(mon) = primary_monitor {
		let mon_pos: [i32; 2] = mon.position().into();
		let mon_size: [u32; 2] = mon.size().into();
		let mon_size_half: IVec2 = (UVec2::from(mon_size) / 2).try_into().unwrap();
		let mon_center = IVec2::from(mon_pos) + mon_size_half;
		let outer_size: [u32; 2] = window.outer_size().into();
		let outer_size_half: IVec2 = (UVec2::from(outer_size) / 2).try_into().unwrap();
		let outer_pos = winit::dpi::Position::Physical((mon_center - outer_size_half).to_array().into());
		window.set_outer_position(outer_pos);
	}

	Ok(Arc::new(window))
}

fn get_configured_present_mode(physical_device: &Arc<PhysicalDevice>, surface: &Surface) -> crate::Result<PresentMode>
{
	let present_mode_regex = regex::Regex::new("--present_mode=(?<value>\\w+)").unwrap();
	let present_mode = std::env::args()
		.collect::<Vec<_>>()
		.iter()
		.find_map(|arg| present_mode_regex.captures(arg))
		.and_then(|caps| caps.name("value"))
		.map_or(PresentMode::Fifo, |value| match value.as_str() {
			"Immediate" => PresentMode::Immediate,
			"Mailbox" => PresentMode::Mailbox,
			"Fifo" => PresentMode::Fifo,
			"FifoRelaxed" => PresentMode::FifoRelaxed,
			_ => PresentMode::Fifo,
		});

	let surface_present_modes: smallvec::SmallVec<[_; 4]> = physical_device
		.surface_present_modes(surface, SurfaceInfo::default())?
		.collect();
	log::info!("Supported present modes: {:?}", &surface_present_modes);

	if surface_present_modes.contains(&present_mode) {
		Ok(present_mode)
	} else {
		log::warn!("Requested present mode `{present_mode:?}` is not supported, falling back to `Fifo`...");
		Ok(PresentMode::Fifo)
	}
}
