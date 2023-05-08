/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::device::{
	physical::{PhysicalDevice, PhysicalDeviceType},
	DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
};

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

fn create_vulkan_instance(game_name: &str) -> Result<Arc<vulkano::instance::Instance>, GenericEngineError>
{
	let lib = vulkano::library::VulkanLibrary::new()?;

	// we'll need to enable the `enumerate_portability` extension if we want to use devices with non-conformant Vulkan
	// implementations like MoltenVK. for now, we can go without it.
	let vk_ext = vulkano_win::required_extensions(&lib);

	// only use the validation layer in debug builds
	#[cfg(debug_assertions)]
	let vk_layers = vec!["VK_LAYER_KHRONOS_validation".into()];
	#[cfg(not(debug_assertions))]
	let vk_layers = Vec::new();

	let mut inst_create_info = vulkano::instance::InstanceCreateInfo::application_from_cargo_toml();
	inst_create_info.application_name = Some(game_name.to_string());
	inst_create_info.engine_name = Some("MithrilEngine".to_string());
	inst_create_info.engine_version = inst_create_info.application_version.clone();
	inst_create_info.enabled_extensions = vk_ext;
	inst_create_info.enabled_layers = vk_layers;
	inst_create_info.max_api_version = Some(vulkano::Version::V1_3);

	Ok(vulkano::instance::Instance::new(lib, inst_create_info)?)
}

/// Get the most appropriate GPU.
fn get_physical_device(vkinst: &Arc<vulkano::instance::Instance>) -> Result<Arc<PhysicalDevice>, GenericEngineError>
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
				dgpu.get_or_insert(pd);
			}
			PhysicalDeviceType::IntegratedGpu => {
				igpu.get_or_insert(pd);
			}
			_ => (),
		}
	}

	// If the "-prefer_igp" argument was provided, prefer the integrated GPU over the discrete GPU.
	let prefer_igp = std::env::args().find(|arg| arg == "-prefer_igp").is_some();

	let physical_device = if prefer_igp {
		igpu.or(dgpu).ok_or("No GPUs were found!")?
	} else {
		// Try to use a discrete GPU. If there is no discrete GPU, use an integrated GPU instead.
		dgpu.or(igpu).ok_or("No GPUs were found!")?
	};
	log::info!("Using physical device: {}", physical_device.properties().device_name);
	Ok(physical_device)
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

		if q.queue_flags.graphics {
			graphics.get_or_insert(i);
		} else if !q.queue_flags.compute && q.queue_flags.transfer {
			transfer_only.get_or_insert(i);
		} else if q.queue_flags.transfer {
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
/// Returns a graphics queue (which owns the device), and an optional transfer queue.
pub fn vulkan_setup(game_name: &str) -> Result<(Arc<Queue>, Option<Arc<Queue>>), GenericEngineError>
{
	let vkinst = create_vulkan_instance(game_name)?;
	let physical_device = get_physical_device(&vkinst)?;

	// The features and extensions enabled here are supported by basically any Vulkan device.
	let enabled_features = vulkano::device::Features {
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
	Ok((graphics_queue, transfer_queue))
}
