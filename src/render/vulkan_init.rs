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

use crate::EngineError;

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
			Self::Nvidia((a, b, c, d)) => write!(f, "{a}.{b}.{c}.{d}"),
			#[cfg(target_family = "windows")]
			Self::IntelWindows((a, b)) => write!(f, "{a}.{b}"),
			Self::Other((a, b, c)) => write!(f, "{a}.{b}.{c}"),
		}
	}
}

fn create_vulkan_instance(
	game_name: &str,
	event_loop: &winit::event_loop::EventLoop<()>,
) -> crate::Result<Arc<vulkano::instance::Instance>>
{
	let lib = vulkano::library::VulkanLibrary::new().map_err(|e| EngineError::new("failed to load Vulkan library", e))?;

	// We'll need to enable the `enumerate_portability` extension if we want to use devices with non-conformant Vulkan
	// implementations like MoltenVK. For now, we can go without it.
	let wanted_extensions = InstanceExtensions {
		ext_swapchain_colorspace: true,
		..Default::default()
	};
	let enabled_extensions = wanted_extensions
		.intersection(lib.supported_extensions())
		.union(&Surface::required_extensions(event_loop));
	log::info!("Enabling instance extensions: {:?}", enabled_extensions);

	// only use the validation layer in debug builds
	#[cfg(debug_assertions)]
	let enabled_layers = vec!["VK_LAYER_KHRONOS_validation".into()];
	#[cfg(not(debug_assertions))]
	let enabled_layers = Vec::new();

	let mut inst_create_info = InstanceCreateInfo {
		application_name: Some(game_name.into()),
		engine_name: Some("MithrilEngine".into()),
		max_api_version: Some(vulkano::Version::V1_3),
		enabled_layers,
		enabled_extensions,
		..InstanceCreateInfo::application_from_cargo_toml()
	};
	inst_create_info.engine_version = inst_create_info.application_version;

	vulkano::instance::Instance::new(lib, inst_create_info)
		.map_err(|e| EngineError::new("failed to create Vulkan instance", e.unwrap()))
}

/// Get the most appropriate GPU, along with whether or not buffers can be directly written to.
fn get_physical_device(vkinst: &Arc<vulkano::instance::Instance>) -> crate::Result<Arc<PhysicalDevice>>
{
	log::info!("Available Vulkan physical devices:");
	let (dgpu, igpu) = vkinst
		.enumerate_physical_devices()
		.map_err(|e| EngineError::new("failed to enumerate physical devices", e))?
		.enumerate()
		.fold((None, None), |(dgpu, igpu), (i, pd)| {
			let properties = pd.properties();
			let driver_ver = DriverVersion::new(properties.driver_version, properties.vendor_id);
			let driver_name = properties.driver_name.as_ref().map_or("unknown driver", |name| name);

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
				PhysicalDeviceType::DiscreteGpu => (dgpu.or(Some((i, pd))), igpu),
				PhysicalDeviceType::IntegratedGpu => (dgpu, igpu.or(Some((i, pd)))),
				_ => (dgpu, igpu),
			}
		});

	// By default, prefer the discrete GPU over the integrated GPU. However, if the "--prefer_igp"
	// argument was provided, instead prefer the integrated GPU over the discrete GPU.
	let (pd_i, physical_device) = std::env::args()
		.find(|arg| arg == "--prefer_igp")
		.and_then(|_| igpu.clone().or_else(|| dgpu.clone()))
		.or_else(|| dgpu.or(igpu))
		.ok_or("No GPUs were found!")?;
	log::info!("Using physical device {}: {}", pd_i, physical_device.properties().device_name);

	// Print all the memory heaps and their types.
	log::info!("Memory heaps and their memory types on physical device:");
	let mem_properties = physical_device.memory_properties();
	for (mem_heap, i) in mem_properties.memory_heaps.iter().zip(0..) {
		let mib = mem_heap.size / (1024 * 1024);
		log::info!("{i}: {mib} MiB, flags {:?}", mem_heap.flags);

		mem_properties
			.memory_types
			.iter()
			.enumerate()
			.filter(|(_, t)| t.heap_index == i)
			.for_each(|(j, mem_type)| log::info!("└ type {j}: {:?}", mem_type.property_flags));
	}

	Ok(physical_device)
}

/// Check if we can directly write to buffer memory on VRAM, as doing so may improve performance.
fn check_direct_buffer_write(physical_device: &Arc<PhysicalDevice>) -> bool
{
	match physical_device.properties().device_type {
		PhysicalDeviceType::DiscreteGpu => {
			// For discrete GPUs, look for a memory type with the property flags
			// `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` belongs to the largest `DEVICE_LOCAL`
			// heap. This is the case when Resizable BAR is enabled, in which case all of the VRAM
			// is host-visible, and writes would immediately go across the PCIe interface.
			let mem_properties = physical_device.memory_properties();
			mem_properties
				.memory_types
				.iter()
				.find(|t| {
					t.property_flags.contains(
						MemoryPropertyFlags::DEVICE_LOCAL
							| MemoryPropertyFlags::HOST_VISIBLE
							| MemoryPropertyFlags::HOST_COHERENT,
					)
				})
				.is_some_and(|t| {
					mem_properties
						.memory_heaps
						.iter()
						.zip(0..)
						.filter(|(heap, _)| heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL))
						.max_by_key(|(heap, _)| heap.size)
						.is_some_and(|(_, i)| t.heap_index == i)
				})
		}

		// For integrated GPUs, directly writing to buffer memory is always possible.
		PhysicalDeviceType::IntegratedGpu => true,

		// For other physical device types, assume we can't directly write to buffer memory.
		_ => false,
	}
}

/// Get a graphics queue family and an optional transfer queue family, then genereate queue create infos for each.
fn get_queue_infos(physical_device: Arc<PhysicalDevice>) -> crate::Result<Vec<QueueCreateInfo>>
{
	let queue_family_properties = physical_device.queue_family_properties();

	log::info!("Available physical device queue families:");
	for (i, q) in queue_family_properties.iter().enumerate() {
		log::info!("{}: {} queue(s), {:?}", i, q.queue_count, q.queue_flags);
	}

	// Get the required graphics queue family, and try to get an optional one for async transfers.
	let (_, graphics) = queue_family_properties
		.iter()
		.zip(0..)
		.find(|(q, _)| q.queue_flags.contains(QueueFlags::GRAPHICS))
		.ok_or("No graphics queue family found!")?;

	// Get another queue family that is ideally specifically optimized for async transfers,
	// by means of finding one with the TRANSFER flag set and least number of flags set.
	let (_, transfer) = queue_family_properties
		.iter()
		.zip(0..)
		.filter(|(q, i)| *i != graphics && q.queue_flags.contains(QueueFlags::TRANSFER))
		.min_by_key(|(q, _)| q.queue_flags.count())
		.unzip();

	if let Some(tq) = transfer {
		log::info!("Using queue family {} for transfers", tq);
	}

	let infos = [graphics]
		.into_iter()
		.chain(transfer)
		.map(|queue_family_index| QueueCreateInfo {
			queue_family_index,
			..Default::default()
		})
		.collect();

	Ok(infos)
}

/// Set up the Vulkan instance, logical device, and queue. Returns a graphics queue (which owns the
/// device) and an optional transfer queue, along with a `bool` indicating whether or not buffers
/// on the device can be directly written to.
pub fn vulkan_setup(
	game_name: &str,
	event_loop: &winit::event_loop::EventLoop<()>,
) -> crate::Result<(Arc<Queue>, Option<Arc<Queue>>, bool)>
{
	let vkinst = create_vulkan_instance(game_name, event_loop)?;
	let physical_device = get_physical_device(&vkinst)?;

	let direct_buffer_write = check_direct_buffer_write(&physical_device);
	if direct_buffer_write {
		log::info!("Enabling direct buffer writes.");
	}

	// The features enabled here are supported by basically any Vulkan device.
	let dev_create_info = DeviceCreateInfo {
		enabled_extensions: DeviceExtensions {
			khr_swapchain: true,
			..Default::default()
		},
		enabled_features: vulkano::device::Features {
			descriptor_binding_variable_descriptor_count: true,
			descriptor_indexing: true,
			dynamic_rendering: true,
			image_cube_array: true,
			independent_blend: true,
			runtime_descriptor_array: true,
			sampler_anisotropy: true,
			shader_sampled_image_array_non_uniform_indexing: true,
			texture_compression_bc: true,
			..Default::default()
		},
		queue_create_infos: get_queue_infos(physical_device.clone())?,
		..Default::default()
	};

	let (_, mut queues) = vulkano::device::Device::new(physical_device, dev_create_info)
		.map_err(|e| EngineError::new("failed to create Vulkan logical device", e.unwrap()))?;
	let graphics_queue = queues.next().unwrap();
	let transfer_queue = queues.next();
	Ok((graphics_queue, transfer_queue, direct_buffer_write))
}
