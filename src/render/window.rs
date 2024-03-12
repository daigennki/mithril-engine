/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use glam::*;
use std::sync::Arc;
use std::time::Duration;
use vulkano::command_buffer::PrimaryCommandBufferAbstract;
use vulkano::device::{
	physical::{PhysicalDevice, PhysicalDeviceType},
	*,
};
use vulkano::format::Format;
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::*;
use vulkano::swapchain::*;
use vulkano::sync::future::{FenceSignalFuture, GpuFuture};
use vulkano::{Validated, Version, VulkanError, VulkanLibrary};
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowBuilder};

use crate::EngineError;

const WINDOW_MIN_INNER_SIZE: winit::dpi::PhysicalSize<u32> = winit::dpi::PhysicalSize::new(1280, 720);

// Surface formats we can support, in order from most preferred to least preferred.
const SURFACE_FORMAT_CANDIDATES: [(Format, ColorSpace); 4] = [
	// HDR via extended sRGB linear image
	// (disabled for now since this is sometimes "supported" on Windows when HDR is disabled for some reason)
	//(Format::R16G16B16A16_SFLOAT, ColorSpace::ExtendedSrgbLinear),

	// sRGB 10bpc
	(Format::A2B10G10R10_UNORM_PACK32, ColorSpace::SrgbNonLinear),
	// sRGB 8bpc (the most widely supported format)
	(Format::B8G8R8A8_UNORM, ColorSpace::SrgbNonLinear),
	// Alternate sRGB 8bpc formats, just to make sure we don't accidentally use their `_SRGB`
	// counterparts when falling back to the first supported format. (the VK_KHR_surface spec states
	// that the `_UNORM` counterpart must also be supported for every support `_SRGB` format)
	(Format::R8G8B8A8_UNORM, ColorSpace::SrgbNonLinear),
	(Format::A8B8G8R8_UNORM_PACK32, ColorSpace::SrgbNonLinear),
];

// The sleep overshoot that gets subtracted from the minimum frame time for the framerate limiter.
// Overshoot differs between Windows and Linux.
#[cfg(target_family = "windows")]
const SLEEP_OVERSHOOT: Duration = Duration::from_micros(260);
#[cfg(not(target_family = "windows"))]
const SLEEP_OVERSHOOT: Duration = Duration::from_micros(50);

pub struct GameWindow
{
	window: Arc<Window>,
	graphics_queue: Arc<Queue>,
	transfer_queue: Option<Arc<Queue>>,
	swapchain: Arc<Swapchain>,
	images: Vec<Arc<Image>>,
	submission_future: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,

	extent_changed: bool, // `true` if image extent changed since the last presentation
	recreate_pending: bool,
	last_frame_presented: std::time::Instant,
	frame_time: Duration,
	frame_time_min_limit: Duration, // minimum frame time, used for framerate limit

	alt_pressed: bool,
}
impl GameWindow
{
	pub fn new(event_loop: &EventLoop<()>, app_name: &str, app_version: Version) -> crate::Result<Self>
	{
		let (graphics_queue, transfer_queue) = vulkan_setup(app_name, app_version, event_loop)?;
		let device = graphics_queue.device();
		let pd = device.physical_device();
		let window = create_window(event_loop, app_name)?;
		let surface = Surface::from_window(device.instance().clone(), window.clone())?;

		// Use the first format candidate supported by the physical device. If none of the format
		// candidates are supported, fall back to the first format returned by `surface_formats`.
		// (the VK_KHR_surface spec states that there must be at least one pair returned)
		let surface_formats = pd.surface_formats(&surface, SurfaceInfo::default())?;
		let (image_format, image_color_space) = SURFACE_FORMAT_CANDIDATES
			.into_iter()
			.find(|candidate| surface_formats.contains(candidate))
			.unwrap_or_else(|| {
				log::warn!(
					"None of the surface format candidates are supported! Falling back to the first supported format,\
					so the image might look weird!",
				);
				// Try to get the first supported sRGB non-linear format, and if that doesn't turn
				// up anything, just get the first one without filtering.
				surface_formats
					.iter()
					.find(|(_, color_space)| *color_space == ColorSpace::SrgbNonLinear)
					.copied()
					.unwrap_or_else(|| surface_formats[0])
			});

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
		let (swapchain, images) = Swapchain::new(device.clone(), surface, create_info)?;

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

		Ok(Self {
			window,
			graphics_queue,
			transfer_queue,
			swapchain,
			images,
			submission_future: None,
			extent_changed: false,
			recreate_pending: false,
			last_frame_presented: std::time::Instant::now(),
			frame_time: Duration::ZERO,
			frame_time_min_limit,
			alt_pressed: false,
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
		cb: Arc<impl PrimaryCommandBufferAbstract + 'static>,
		acquire_future: SwapchainAcquireFuture,
		join_with: Option<impl GpuFuture + Send + Sync + 'static>,
	) -> crate::Result<()>
	{
		let queue = self.graphics_queue.clone();
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

	pub fn graphics_queue(&self) -> &Arc<Queue>
	{
		&self.graphics_queue
	}
	pub fn transfer_queue(&self) -> &Option<Arc<Queue>>
	{
		&self.transfer_queue
	}

	pub fn set_fullscreen(&self, fullscreen: bool)
	{
		self.window.set_fullscreen(fullscreen.then_some(Fullscreen::Borderless(None)));
	}
	pub fn is_fullscreen(&self) -> bool
	{
		self.window.fullscreen().is_some()
	}

	pub fn handle_window_event(&mut self, window_event: &mut WindowEvent)
	{
		match window_event {
			WindowEvent::ScaleFactorChanged { inner_size_writer, .. } => {
				// Reset minimum inner size because it seems to get changed with DPI scale factor changes.
				self.window.set_min_inner_size(Some(WINDOW_MIN_INNER_SIZE));

				// We don't want the image to be upscaled by the OS, so we tell it here that the
				// inner size of the window in physical pixels should be exactly the same
				// (dot-by-dot) as the swapchain's image extent. It would look blurry otherwise.
				let extent = self.swapchain.image_extent();
				if let Err(e) = inner_size_writer.request_inner_size(extent.into()) {
					log::error!("failed to request window inner size: {e}");
				}
			}
			WindowEvent::KeyboardInput {
				event:
					KeyEvent {
						physical_key: PhysicalKey::Code(KeyCode::Enter),
						state: ElementState::Released,
						repeat: false,
						..
					},
				..
			} => {
				if self.alt_pressed {
					self.set_fullscreen(!self.is_fullscreen()); // Toggle fullscreen
				}
			}
			WindowEvent::ModifiersChanged(modifiers) => {
				self.alt_pressed = modifiers.state().contains(ModifiersState::ALT);
			}
			_ => (),
		}
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

//
/* Vulkan initialization */
//
fn get_physical_device(
	app_name: &str,
	application_version: Version,
	event_loop: &EventLoop<()>,
) -> crate::Result<Arc<PhysicalDevice>>
{
	let lib = VulkanLibrary::new().map_err(|e| EngineError::new("failed to load Vulkan library", e))?;

	// only use the validation layer in debug builds
	#[cfg(debug_assertions)]
	let enabled_layers = vec!["VK_LAYER_KHRONOS_validation".into()];
	#[cfg(not(debug_assertions))]
	let enabled_layers = Vec::new();

	const OPTIONAL_INSTANCE_EXTENSIONS: InstanceExtensions = InstanceExtensions {
		ext_swapchain_colorspace: true,
		..InstanceExtensions::empty()
	};
	let inst_create_info = InstanceCreateInfo {
		flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
		application_name: Some(app_name.into()),
		application_version,
		engine_name: Some(env!("CARGO_PKG_NAME").into()),
		engine_version: Version {
			major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
			minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
			patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
		},
		enabled_layers,
		enabled_extensions: OPTIONAL_INSTANCE_EXTENSIONS
			.intersection(lib.supported_extensions())
			.union(&Surface::required_extensions(event_loop)),
		..Default::default()
	};

	let physical_devices: smallvec::SmallVec<[_; 4]> = Instance::new(lib, inst_create_info)?
		.enumerate_physical_devices()
		.map_err(|e| EngineError::new("failed to enumerate physical devices", e))?
		.collect();

	log::info!("Available physical devices:");
	for (i, pd) in physical_devices.iter().enumerate() {
		let device_name = &pd.properties().device_name;
		let driver_name = pd.properties().driver_name.as_ref().map_or("[no driver name]", |name| name);
		let driver_info = pd.properties().driver_info.as_ref().map_or("[no driver info]", |name| name);
		log::info!("{i}: {device_name} (driver: {driver_name}, {driver_info})");
	}

	// Look for discrete and integrated GPUs. Also, for each type of GPU, make sure it actually has
	// a graphics queue family.
	let dgpu_i = physical_devices
		.iter()
		.enumerate()
		.filter(|(_, pd)| {
			pd.queue_family_properties()
				.iter()
				.any(|qf| qf.queue_flags.contains(QueueFlags::GRAPHICS))
		})
		.find_map(|(i, pd)| (pd.properties().device_type == PhysicalDeviceType::DiscreteGpu).then_some(i));
	let igpu_i = physical_devices
		.iter()
		.enumerate()
		.filter(|(_, pd)| {
			pd.queue_family_properties()
				.iter()
				.any(|qf| qf.queue_flags.contains(QueueFlags::GRAPHICS))
		})
		.find_map(|(i, pd)| (pd.properties().device_type == PhysicalDeviceType::IntegratedGpu).then_some(i));

	// Prefer an integrated GPU if the "--prefer_igp" argument was provided. Otherwise, prefer a
	// discrete GPU. If neither a dGPU nor an iGPU were found, use the first physical device with a
	// graphics queue family.
	let pd_i = std::env::args()
		.find(|arg| arg == "--prefer_igp")
		.and_then(|_| igpu_i.or(dgpu_i))
		.or_else(|| dgpu_i.or(igpu_i))
		.or_else(|| {
			physical_devices.iter().position(|pd| {
				pd.queue_family_properties()
					.iter()
					.any(|qf| qf.queue_flags.contains(QueueFlags::GRAPHICS))
			})
		})
		.ok_or("No GPUs with a graphics queue family were found!")?;
	let physical_device = physical_devices.into_iter().nth(pd_i).unwrap();

	let pd_api_ver = physical_device.properties().api_version;
	log::info!("Using physical device {pd_i} (Vulkan {pd_api_ver})");

	Ok(physical_device)
}

// The features enabled here are supported by basically any Vulkan device on PC.
const ENABLED_FEATURES: Features = Features {
	descriptor_binding_variable_descriptor_count: true,
	descriptor_indexing: true,
	dynamic_rendering: true,
	image_cube_array: true,
	inline_uniform_block: true,
	independent_blend: true,
	runtime_descriptor_array: true,
	sampler_anisotropy: true,
	shader_sampled_image_array_non_uniform_indexing: true,
	shader_storage_image_multisample: true,
	texture_compression_bc: true,
	..Features::empty()
};

/// Create the Vulkan logical device. Returns a graphics queue (which owns the device) and an
/// optional transfer queue.
fn vulkan_setup(
	app_name: &str,
	app_version: Version,
	event_loop: &EventLoop<()>,
) -> crate::Result<(Arc<Queue>, Option<Arc<Queue>>)>
{
	let physical_device = get_physical_device(app_name, app_version, event_loop)?;

	let queue_family_properties = physical_device.queue_family_properties();
	for (i, q) in queue_family_properties.iter().enumerate() {
		log::info!("Queue family {i}: {} queue(s), {:?}", q.queue_count, q.queue_flags);
	}

	// Look for a graphics queue family and an optional (preferably dedicated) transfer queue
	// family, and generate `QueueCreateInfo`s for one queue on each.
	let (_, graphics_qfi) = queue_family_properties
		.iter()
		.zip(0..)
		.find(|(q, _)| q.queue_flags.contains(QueueFlags::GRAPHICS))
		.unwrap();

	// For transfers, try to find a queue family with the TRANSFER flag and least number of flags.
	let (_, transfer_qfi) = queue_family_properties
		.iter()
		.zip(0..)
		.filter(|(q, i)| *i != graphics_qfi && q.queue_flags.contains(QueueFlags::TRANSFER))
		.min_by_key(|(q, _)| q.queue_flags.count())
		.unzip();
	if let Some(tq) = transfer_qfi {
		log::info!("Using queue family {} for transfers", tq);
	}

	let queue_create_infos = [graphics_qfi]
		.into_iter()
		.chain(transfer_qfi)
		.map(|queue_family_index| QueueCreateInfo {
			queue_family_index,
			..Default::default()
		})
		.collect();

	let dev_info = DeviceCreateInfo {
		enabled_extensions: DeviceExtensions {
			khr_swapchain: true,
			..Default::default()
		},
		enabled_features: ENABLED_FEATURES,
		queue_create_infos,
		..Default::default()
	};
	let (_, mut queues) = Device::new(physical_device, dev_info)
		.map_err(|e| EngineError::new("failed to create Vulkan logical device", e.unwrap()))?;

	let graphics_queue = queues.next().unwrap();
	let transfer_queue = queues.next();
	Ok((graphics_queue, transfer_queue))
}
