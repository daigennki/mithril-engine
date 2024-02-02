/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;
use vulkano::device::{
	physical::{PhysicalDevice, PhysicalDeviceType},
	Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::library::VulkanLibrary;
use vulkano::memory::MemoryPropertyFlags;
use vulkano::swapchain::Surface;
use vulkano::DeviceSize;
use winit::event_loop::EventLoop;

use crate::EngineError;

struct DriverVersion
{
	version: u32,
	vendor_id: u32,
}
impl std::fmt::Display for DriverVersion
{
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
	{
		let version = self.version;
		match self.vendor_id {
			// NVIDIA
			4318 => write!(
				f,
				"{}.{}.{}.{}",
				version >> 22,
				(version >> 14) & 0x0ff,
				(version >> 6) & 0x0ff,
				version & 0x003f,
			),

			// Intel (Windows only)
			#[cfg(target_family = "windows")]
			0x8086 => write!(f, "{}.{}", version >> 14, version & 0x3fff),

			// Others (use Vulkan version convention)
			_ => vulkano::Version::from(version).fmt(f),
		}
	}
}

fn get_physical_device(vkinst: &Arc<Instance>) -> crate::Result<Arc<PhysicalDevice>>
{
	let physical_devices: smallvec::SmallVec<[_; 4]> = vkinst
		.enumerate_physical_devices()
		.map_err(|e| EngineError::new("failed to enumerate physical devices", e))?
		.collect();

	log::info!("Available Vulkan physical devices:");
	for (i, pd) in physical_devices.iter().enumerate() {
		let device_name = &pd.properties().device_name;
		let device_type = pd.properties().device_type;
		let driver_name = pd.properties().driver_name.as_ref().map_or("[unknown]", |name| name);
		let driver_ver = DriverVersion {
			version: pd.properties().driver_version,
			vendor_id: pd.properties().vendor_id,
		};
		log::info!("{i}: {device_name} ({device_type:?}) (driver: {driver_name} {driver_ver})");
	}

	// Prefer an integrated GPU if the "--prefer_igp" argument was provided. Otherwise, prefer a
	// discrete GPU.
	let dgpu_i = physical_devices
		.iter()
		.position(|pd| pd.properties().device_type == PhysicalDeviceType::DiscreteGpu);
	let igpu_i = physical_devices
		.iter()
		.position(|pd| pd.properties().device_type == PhysicalDeviceType::IntegratedGpu);
	let pd_i = std::env::args()
		.find(|arg| arg == "--prefer_igp")
		.and_then(|_| igpu_i.or(dgpu_i))
		.or_else(|| dgpu_i.or(igpu_i))
		.ok_or("No GPUs were found!")?;
	let physical_device = physical_devices.into_iter().nth(pd_i).unwrap();

	let pd_api_ver = physical_device.properties().api_version;
	log::info!("Using physical device {pd_i} (Vulkan {pd_api_ver})");

	let mem_properties = physical_device.memory_properties();
	for (i, mem_heap) in mem_properties.memory_heaps.iter().enumerate() {
		let mib = mem_heap.size / (1024 * 1024);
		log::info!("Memory heap {i}: {mib} MiB");
	}
	for (i, mem_type) in mem_properties.memory_types.iter().enumerate() {
		log::info!("Memory type {i}: heap {}, {:?}", mem_type.heap_index, mem_type.property_flags);
	}

	Ok(physical_device)
}

// The features enabled here are supported by basically any Vulkan device on PC.
const ENABLED_FEATURES: Features = Features {
	descriptor_binding_variable_descriptor_count: true,
	descriptor_indexing: true,
	dynamic_rendering: true,
	image_cube_array: true,
	independent_blend: true,
	runtime_descriptor_array: true,
	sampler_anisotropy: true,
	shader_sampled_image_array_non_uniform_indexing: true,
	texture_compression_bc: true,
	..Features::empty()
};

fn create_logical_device(physical_device: Arc<PhysicalDevice>) -> crate::Result<(Arc<Queue>, Option<Arc<Queue>>)>
{
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
		.ok_or("No graphics queue family found!")?;

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

/// Set up the Vulkan instance, logical device, and queue. Returns a graphics queue (which owns the
/// device) and an optional transfer queue, along with a `bool` indicating whether or not buffers
/// on the device can be directly written to.
pub fn vulkan_setup(app_name: &str, event_loop: &EventLoop<()>) -> crate::Result<(Arc<Queue>, Option<Arc<Queue>>, bool)>
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
	// TODO: take app version as parameter
	let mut inst_create_info = InstanceCreateInfo {
		application_name: Some(app_name.into()),
		engine_name: Some(env!("CARGO_PKG_NAME").into()),
		engine_version: vulkano::Version {
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
	inst_create_info.engine_version = inst_create_info.application_version;

	log::info!("Enabling instance extensions: {:?}", &inst_create_info.enabled_extensions);

	let vkinst =
		Instance::new(lib, inst_create_info).map_err(|e| EngineError::new("failed to create Vulkan instance", e.unwrap()))?;

	let physical_device = get_physical_device(&vkinst)?;

	// Check if we can directly write to device-local buffer memory, as doing so may be faster.
	let direct_buffer_write = match physical_device.properties().device_type {
		PhysicalDeviceType::IntegratedGpu => true, // Always possible for integrated GPU.
		PhysicalDeviceType::DiscreteGpu => {
			// For discrete GPUs, look for a host-visible memory type belonging to a device-local
			// heap larger than **exactly** 256 **MiB**.
			const DIRECT_WRITE_THRESHOLD: DeviceSize = 256 * 1024 * 1024;
			let mem_properties = physical_device.memory_properties();
			mem_properties
				.memory_types
				.iter()
				.filter(|t| {
					t.property_flags.contains(
						MemoryPropertyFlags::DEVICE_LOCAL
							| MemoryPropertyFlags::HOST_VISIBLE
							| MemoryPropertyFlags::HOST_COHERENT,
					)
				})
				.any(|t| mem_properties.memory_heaps[t.heap_index as usize].size > DIRECT_WRITE_THRESHOLD)
		}
		_ => unreachable!(),
	};
	if direct_buffer_write {
		log::info!("Enabling direct buffer writes.");
	}

	let (graphics_queue, transfer_queue) = create_logical_device(physical_device)?;
	Ok((graphics_queue, transfer_queue, direct_buffer_write))
}
