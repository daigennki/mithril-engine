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
use vulkano::instance::InstanceCreateInfo;
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
	event_loop: &winit::event_loop::EventLoop<()>
) -> Result<Arc<vulkano::instance::Instance>, GenericEngineError>
{
	let lib = vulkano::library::VulkanLibrary::new()?;

	// we'll need to enable the `enumerate_portability` extension if we want to use devices with non-conformant Vulkan
	// implementations like MoltenVK. for now, we can go without it.
	let enabled_extensions = Surface::required_extensions(event_loop);

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
fn get_physical_device(
	vkinst: &Arc<vulkano::instance::Instance>
) -> Result<(Arc<PhysicalDevice>, bool), GenericEngineError>
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

	// If the "-prefer_igp" argument was provided, prefer the integrated GPU over the discrete GPU.
	let prefer_igp = std::env::args().find(|arg| arg == "-prefer_igp").is_some();

	let (i, physical_device) = if prefer_igp {
		igpu.or(dgpu).ok_or("No GPUs were found!")?
	} else {
		// Try to use a discrete GPU. If there is no discrete GPU, use an integrated GPU instead.
		dgpu.or(igpu).ok_or("No GPUs were found!")?
	};
	log::info!("Using physical device {}: {}", i, physical_device.properties().device_name);

	let mem_properties = physical_device.memory_properties();

	// Query for Resizable BAR support by checking if the largest `DEVICE_LOCAL` heap also has `HOST_VISIBLE`.
	// If Resizable BAR is enabled, we can initialize buffers directly on the GPU, which may improve performance.
	let (largest_dev_local_i, _) = mem_properties
		.memory_heaps
		.iter()
		.enumerate()
		.filter(|(_, heap)| heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL))
		.max_by_key(|(_, heap)| heap.size)
		.unwrap();
	let rebar_in_use = mem_properties
		.memory_types
		.iter()
		.filter(|t| t.heap_index as usize == largest_dev_local_i)
		.find(|t| t.property_flags.contains(MemoryPropertyFlags::HOST_VISIBLE))
		.is_some();

	// Print all the memory heaps and their types.
	log::info!("Memory heaps and their memory types on physical device:");
	for (i, mem_heap) in mem_properties.memory_heaps.iter().enumerate() {
		let rebar_text = (i == largest_dev_local_i && rebar_in_use)
			.then_some("yes")
			.unwrap_or("no");

		let mib = mem_heap.size / (1024 * 1024);

		log::info!("{}: {} MiB, flags {:?} (Resizable BAR: {})", i, mib, mem_heap.flags, rebar_text);

		for mem_type in mem_properties.memory_types.iter().filter(|t| t.heap_index as usize == i) {
			log::info!("└ {:?}", mem_type.property_flags);
		}
	}
	Ok((physical_device, rebar_in_use))
}

/// Get a graphics queue family and an optional transfer queue family, then genereate queue create infos for each.
fn get_queue_infos(physical_device: Arc<PhysicalDevice>) -> Result<Vec<QueueCreateInfo>, GenericEngineError>
{
	let mut graphics = None; // required
	let mut transfer_only = None; // optional; optimized specifically for transfers
	let mut transfer = None; // optional; not transfer-specific, but still works for async transfers

	// Get the required graphics queue family, and try to get an optional one for async transfers.
	// For transfers, try to get one that is specifically optimized for async transfers (supports netiher graphics nor compute),
	// then if such a queue family doesn't exist, use one that just doesn't support graphics.
	log::info!("Available physical device queue families:");
	for (i, q) in physical_device.queue_family_properties().iter().enumerate() {
		log::info!("{}: {} queue(s), {:?}", i, q.queue_count, q.queue_flags);

		if q.queue_flags.intersects(QueueFlags::GRAPHICS) {
			graphics.get_or_insert(i);
		} else if !q.queue_flags.intersects(QueueFlags::COMPUTE) && q.queue_flags.intersects(QueueFlags::TRANSFER) {
			transfer_only.get_or_insert(i);
		} else if q.queue_flags.intersects(QueueFlags::TRANSFER) {
			transfer.get_or_insert(i);
		}
	}

	let mut use_queue_families = vec![graphics.ok_or("No graphics queue family found!")?];
	if let Some(tq) = transfer_only.or(transfer) {
		log::info!("Using queue family {} for transfers", tq);
		use_queue_families.push(tq);
	}

	let infos = use_queue_families
		.into_iter()
		.map(|i| QueueCreateInfo {
			queue_family_index: i as u32,
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
	event_loop: &winit::event_loop::EventLoop<()>
) -> Result<(Arc<Queue>, Option<Arc<Queue>>, bool), GenericEngineError>
{
	let vkinst = create_vulkan_instance(game_name, event_loop)?;
	let (physical_device, rebar_in_use) = get_physical_device(&vkinst)?;

	// The features and extensions enabled here are supported by basically any Vulkan device.
	let enabled_features = vulkano::device::Features {
		dynamic_rendering: true,
		image_cube_array: true,
		independent_blend: true,
		sampler_anisotropy: true,
		texture_compression_bc: true,
		geometry_shader: true,
		..Default::default()
	};

	let dev_create_info = DeviceCreateInfo {
		enabled_extensions: DeviceExtensions {
			khr_swapchain: true,
			..Default::default()
		},
		enabled_features,
		queue_create_infos: get_queue_infos(physical_device.clone())?,
		..Default::default()
	};

	let (_, mut queues) = vulkano::device::Device::new(physical_device, dev_create_info)?;
	let graphics_queue = queues.next().unwrap();
	let transfer_queue = queues.next();
	Ok((graphics_queue, transfer_queue, rebar_in_use))
}
