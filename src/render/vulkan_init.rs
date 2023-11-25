/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;
use vulkano::device::{
	physical::{PhysicalDevice, PhysicalDeviceType},
	DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::{MemoryHeapFlags, MemoryPropertyFlags};
use vulkano::swapchain::Surface;

use crate::GenericEngineError;

enum DriverVersion
{
	Nvidia((u32, u32, u32, u32)),
	#[cfg(target_family = "windows")]
	IntelWindows((u32, u32)),
	Other((u32, u32, u32)),
}
impl DriverVersion
{
	fn new(version: u32, vendor_id: u32) -> Self
	{
		// NVIDIA
		if vendor_id == 4318 {
			return Self::Nvidia((
				(version >> 22) & 0x3ff,
				(version >> 14) & 0x0ff,
				(version >> 6) & 0x0ff,
				version & 0x003f,
			));
		}

		// Intel (Windows only)
		#[cfg(target_family = "windows")]
		if vendor_id == 0x8086 {
			return Self::IntelWindows(((version >> 14), version & 0x3fff));
		}

		// others (use Vulkan version convention)
		Self::Other(((version >> 22), (version >> 12) & 0x3ff, version & 0xfff))
	}
}
impl std::fmt::Display for DriverVersion
{
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
	{
		match self {
			Self::Nvidia((a, b, c, d)) => write!(f, "{}.{}.{}.{}", a, b, c, d),
			#[cfg(target_family = "windows")]
			Self::IntelWindows((a, b)) => write!(f, "{}.{}", a, b),
			Self::Other((a, b, c)) => write!(f, "{}.{}.{}", a, b, c),
		}
	}
}

fn create_vulkan_instance(
	game_name: &str,
	event_loop: &winit::event_loop::EventLoop<()>,
) -> Result<Arc<vulkano::instance::Instance>, GenericEngineError>
{
	let lib = vulkano::library::VulkanLibrary::new()?;

	// We'll need to enable the `enumerate_portability` extension if we want to use devices with non-conformant Vulkan
	// implementations like MoltenVK. For now, we can go without it.
	let wanted_extensions = InstanceExtensions {
		khr_get_surface_capabilities2: true,
		ext_swapchain_colorspace: true,
		..Default::default()
	};
	let enabled_extensions = wanted_extensions
		.intersection(lib.supported_extensions())
		.union(&Surface::required_extensions(event_loop));

	log::info!("Enabling instance extensions:");
	enabled_extensions
		.into_iter()
		.filter_map(|(name, enabled)| enabled.then_some(name))
		.for_each(|ext| log::info!("- {}", ext));

	// only use the validation layer in debug builds
	#[cfg(debug_assertions)]
	let enabled_layers = vec!["VK_LAYER_KHRONOS_validation".into()];
	#[cfg(not(debug_assertions))]
	let enabled_layers = Vec::new();

	let mut inst_create_info = InstanceCreateInfo {
		application_name: Some(game_name.to_string()),
		engine_name: Some("MithrilEngine".to_string()),
		max_api_version: Some(vulkano::Version::V1_3),
		enabled_layers,
		enabled_extensions,
		..InstanceCreateInfo::application_from_cargo_toml()
	};
	inst_create_info.engine_version = inst_create_info.application_version.clone();

	Ok(vulkano::instance::Instance::new(lib, inst_create_info)?)
}

/// Get the most appropriate GPU, along with whether or not Resizable BAR is enabled on its largest `DEVICE_LOCAL` memory heap.
fn get_physical_device(vkinst: &Arc<vulkano::instance::Instance>) -> Result<(Arc<PhysicalDevice>, bool), GenericEngineError>
{
	log::info!("Available Vulkan physical devices:");
	let (mut dgpu, mut igpu) = (None, None);
	for (i, pd) in vkinst.enumerate_physical_devices()?.enumerate() {
		let properties = pd.properties();
		let driver_ver = DriverVersion::new(properties.driver_version, properties.vendor_id);
		let driver_name = properties.driver_name.as_ref().map_or("unknown driver", |name| &name);

		log::info!(
			"{}: {} ({:?}), driver '{}' version {} (Vulkan {})",
			i,
			properties.device_name,
			properties.device_type,
			driver_name,
			driver_ver,
			properties.api_version
		);

		match properties.device_type {
			PhysicalDeviceType::DiscreteGpu => {
				dgpu.get_or_insert((i, pd));
			}
			PhysicalDeviceType::IntegratedGpu => {
				igpu.get_or_insert((i, pd));
			}
			_ => (),
		}
	}

	// If the "--prefer_igp" argument was provided, prefer the integrated GPU over the discrete GPU.
	let prefer_igp = std::env::args().find(|arg| arg == "--prefer_igp").is_some();

	let (i, physical_device) = if prefer_igp {
		igpu.or(dgpu).ok_or("No GPUs were found!")?
	} else {
		// Try to use a discrete GPU. If there is no discrete GPU, use an integrated GPU instead.
		dgpu.or(igpu).ok_or("No GPUs were found!")?
	};
	log::info!("Using physical device {}: {}", i, physical_device.properties().device_name);

	let mem_properties = physical_device.memory_properties();
	let device_type = physical_device.properties().device_type;

	// Check if we can write to buffer memory on VRAM directly, as doing so may improve performance.
	let allow_direct_buffer_access;
	match device_type {
		PhysicalDeviceType::DiscreteGpu => {
			// For discrete GPUs, check that the largest `DEVICE_LOCAL` memory heap is also `HOST_VISIBLE`.
			// This is the case when Resizable BAR is enabled.
			let (_, largest_heap_i) = mem_properties
				.memory_heaps
				.iter()
				.zip(0_u32..)
				.filter(|(heap, _)| heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL))
				.max_by_key(|(heap, _)| heap.size)
				.ok_or("`DiscreteGpu` doesn't have any `DEVICE_LOCAL` memory heaps!")?;

			allow_direct_buffer_access = mem_properties
				.memory_types
				.iter()
				.filter(|t| t.heap_index == largest_heap_i)
				.any(|t| t.property_flags.contains(MemoryPropertyFlags::HOST_VISIBLE));

			if allow_direct_buffer_access {
				log::info!("Resizable BAR appears to be enabled on this physical device.");
			}
		}
		PhysicalDeviceType::IntegratedGpu => {
			// For integrated GPUs, assume that writing directly to buffer memory is always possible and fast enough.
			allow_direct_buffer_access = true;
		}
		_ => {
			// For other physical device types, assume we can't write directly to buffer memory.
			allow_direct_buffer_access = false;
		}
	}

	// Print all the memory heaps and their types.
	log::info!("Memory heaps and their memory types on physical device:");
	for (mem_heap, i) in mem_properties.memory_heaps.iter().zip(0_u32..) {
		let mib = mem_heap.size / (1024 * 1024);

		log::info!("{}: {} MiB, flags {:?}", i, mib, mem_heap.flags);

		for mem_type in mem_properties.memory_types.iter().filter(|t| t.heap_index == i) {
			log::info!("└ {:?}", mem_type.property_flags);
		}
	}
	Ok((physical_device, allow_direct_buffer_access))
}

/// Get a graphics queue family and an optional transfer queue family, then genereate queue create infos for each.
fn get_queue_infos(physical_device: Arc<PhysicalDevice>) -> Result<Vec<QueueCreateInfo>, GenericEngineError>
{
	let queue_family_properties = physical_device.queue_family_properties();

	log::info!("Available physical device queue families:");
	for (i, q) in queue_family_properties.iter().enumerate() {
		log::info!("{}: {} queue(s), {:?}", i, q.queue_count, q.queue_flags);
	}

	// Get the required graphics queue family, and try to get an optional one for async transfers.
	let graphics = queue_family_properties
		.iter()
		.zip(0_u32..)
		.find_map(|(q, i)| q.queue_flags.contains(QueueFlags::GRAPHICS).then_some(i))
		.ok_or("No graphics queue family found!")?;

	// Get another queue family that is ideally specifically optimized for async transfers,
	// by means of finding one with the TRANSFER flag set and least number of flags set.
	let transfer = queue_family_properties
		.iter()
		.zip(0_u32..)
		.filter(|(q, i)| *i != graphics && q.queue_flags.contains(QueueFlags::TRANSFER))
		.min_by_key(|(q, _)| q.queue_flags.count())
		.map(|(_, i)| i);

	if let Some(tq) = transfer {
		log::info!("Using queue family {} for transfers", tq);
	}

	let infos = [graphics]
		.into_iter()
		.chain(transfer.into_iter())
		.map(|queue_family_index| QueueCreateInfo {
			queue_family_index,
			..Default::default()
		})
		.collect();

	Ok(infos)
}

/// Set up the Vulkan instance, physical device, logical device, and queue.
/// Returns a graphics queue (which owns the device), and an optional transfer queue,
/// along with whether or not Resizable BAR is in use.
pub fn vulkan_setup(
	game_name: &str,
	event_loop: &winit::event_loop::EventLoop<()>,
) -> Result<(Arc<Queue>, Option<Arc<Queue>>, bool), GenericEngineError>
{
	let vkinst = create_vulkan_instance(game_name, event_loop)?;
	let (physical_device, rebar_in_use) = get_physical_device(&vkinst)?;

	let wanted_extensions = DeviceExtensions {
		// currently no device extensions enabled; leaving this struct here for convenience
		..Default::default()
	};
	let enabled_extensions = DeviceExtensions {
		khr_swapchain: true,
		..wanted_extensions.intersection(physical_device.supported_extensions())
	};
	log::info!("Enabling device extensions:");
	enabled_extensions
		.into_iter()
		.filter_map(|(name, enabled)| enabled.then_some(name))
		.for_each(|ext| log::info!("- {}", ext));

	// The features enabled here are supported by basically any Vulkan device.
	let enabled_features = vulkano::device::Features {
		dynamic_rendering: true,
		image_cube_array: true,
		independent_blend: true,
		sampler_anisotropy: true,
		texture_compression_bc: true,
		..Default::default()
	};

	let dev_create_info = DeviceCreateInfo {
		enabled_extensions,
		enabled_features,
		queue_create_infos: get_queue_infos(physical_device.clone())?,
		..Default::default()
	};

	let (_, mut queues) = vulkano::device::Device::new(physical_device, dev_create_info)?;
	let graphics_queue = queues.next().unwrap();
	let transfer_queue = queues.next();
	Ok((graphics_queue, transfer_queue, rebar_in_use))
}
