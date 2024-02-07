/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod lighting;
pub mod model;
mod skybox;
mod swapchain;
mod transfer;
mod transparency;
pub mod ui;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddsfile::DxgiFormat;
use glam::*;
use shipyard::{IntoWorkload, UniqueView, UniqueViewMut, Workload};

use vulkano::buffer::{subbuffer::Subbuffer, Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{
	allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{
	physical::{PhysicalDevice, PhysicalDeviceType},
	Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::{Format, FormatFeatures};
use vulkano::image::{view::ImageView, Image, ImageCreateInfo, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::{
	allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
	MemoryPropertyFlags,
};
use vulkano::pipeline::{
	compute::ComputePipelineCreateInfo, layout::PipelineLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint,
	PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::ShaderStages;
use vulkano::swapchain::{ColorSpace, Surface};
use vulkano::{DeviceSize, VulkanLibrary};
use winit::event_loop::EventLoop;

use crate::component::camera::CameraManager;
use crate::EngineError;
use lighting::LightManager;
use model::MeshManager;
use ui::Canvas;

// Some notes regarding observed support (with AMD, Intel, and NVIDIA GPUs) for depth/stencil formats:
//
// - `D16_UNORM`: Supported on all GPUs.
// - `D16_UNORM_S8_UINT`: Only supported on AMD GPUs.
// - `X8_D24_UNORM_PACK32`: Only supported on NVIDIA and Intel GPUs.
// - `D24_UNORM_S8_UINT`: Only supported on NVIDIA and Intel GPUs.
// - `D32_SFLOAT`: Supported on all GPUs.
// - `D32_SFLOAT_S8_UINT`: Supported on all GPUs.
// - `S8_UINT`: Only supported on AMD GPUs. Possibly supported on NVIDIA GPUs.
//
// (source: https://vulkan.gpuinfo.org/listoptimaltilingformats.php)
const DEPTH_STENCIL_FORMAT_CANDIDATES: [Format; 2] = [Format::D24_UNORM_S8_UINT, Format::D16_UNORM_S8_UINT];

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	graphics_queue: Arc<Queue>,
	memory_allocator: Arc<StandardMemoryAllocator>,
	command_buffer_allocator: StandardCommandBufferAllocator,
	direct_buffer_write: bool,
	transfer_manager: transfer::TransferManager,
	swapchain: swapchain::Swapchain,
	skybox: Option<skybox::Skybox>,
	transparency_renderer: Option<transparency::MomentTransparencyRenderer>,

	// Things related to the output color/depth images and gamma correction.
	color_set_allocator: StandardDescriptorSetAllocator,
	color_set_layout: Arc<DescriptorSetLayout>,
	depth_stencil_format: Format,
	color_image: Option<Arc<ImageView>>,
	depth_image: Option<Arc<ImageView>>,
	color_set: Option<Arc<PersistentDescriptorSet>>, // Contains `color_image` as a storage image binding.
	gamma_pipeline: Arc<ComputePipeline>,            // The gamma correction pipeline.

	// Loaded textures, with the key being the path relative to the current working directory.
	textures: HashMap<PathBuf, Arc<ImageView>>,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &EventLoop<()>) -> crate::Result<Self>
	{
		let (graphics_queue, transfer_queue) = vulkan_setup(game_name, event_loop)?;
		let vk_dev = graphics_queue.device().clone();
		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_dev.clone()));

		// Check if we can directly write to device-local buffer memory, as doing so may be faster.
		let direct_buffer_write = check_direct_buffer_write(vk_dev.physical_device());
		if direct_buffer_write {
			log::info!("Enabling direct buffer writes.");
		}

		let swapchain = swapchain::Swapchain::new(vk_dev.clone(), event_loop, game_name)?;

		// - Primary command buffers: One for each graphics submission.
		// - Secondary command buffers: Only up to four should be created per thread.
		let cb_alloc_info = StandardCommandBufferAllocatorCreateInfo {
			primary_buffer_count: swapchain.image_count(),
			secondary_buffer_count: 4,
			..Default::default()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(vk_dev.clone(), cb_alloc_info);

		let submit_transfers_to = transfer_queue.unwrap_or_else(|| graphics_queue.clone());
		let transfer_manager = transfer::TransferManager::new(submit_transfers_to, memory_allocator.clone())?;

		let color_set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 2,
			..Default::default()
		};
		let color_set_allocator = StandardDescriptorSetAllocator::new(vk_dev.clone(), color_set_alloc_info);

		let depth_stencil_format = DEPTH_STENCIL_FORMAT_CANDIDATES
			.into_iter()
			.find(|format| {
				vk_dev
					.physical_device()
					.format_properties(*format)
					.unwrap()
					.optimal_tiling_features
					.contains(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
			})
			.ok_or("none of the depth/stencil format candidates are supported")?;
		log::debug!("using depth/stencil format {depth_stencil_format:?}");

		let color_image_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::COMPUTE,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
		};
		let color_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [(0, color_image_binding)].into(),
			..Default::default()
		};
		let color_set_layout = DescriptorSetLayout::new(vk_dev.clone(), color_set_layout_info)?;

		let pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![color_set_layout.clone()],
			..Default::default()
		};
		let pipeline_layout = PipelineLayout::new(vk_dev.clone(), pipeline_layout_info)?;
		let entry_point = compute_gamma::load(vk_dev.clone())?.entry_point("main").unwrap();
		let stage = PipelineShaderStageCreateInfo::new(entry_point);
		let pipeline_create_info = ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout);
		let gamma_pipeline = ComputePipeline::new(vk_dev.clone(), None, pipeline_create_info)?;

		Ok(Self {
			graphics_queue,
			memory_allocator,
			command_buffer_allocator,
			direct_buffer_write,
			transfer_manager,
			swapchain,
			skybox: None,
			transparency_renderer: None,
			color_set_allocator,
			color_set_layout,
			depth_stencil_format,
			color_image: None,
			depth_image: None,
			color_set: None,
			gamma_pipeline,
			textures: HashMap::new(),
		})
	}

	pub fn set_skybox(&mut self, path_pattern: String) -> crate::Result<()>
	{
		self.skybox = Some(skybox::Skybox::new(self, path_pattern)?);
		Ok(())
	}

	fn load_transparency(&mut self, material_textures_set_layout: Arc<DescriptorSetLayout>) -> crate::Result<()>
	{
		self.transparency_renderer = Some(transparency::MomentTransparencyRenderer::new(
			self.memory_allocator.clone(),
			material_textures_set_layout,
			self.swapchain.dimensions(),
			self.depth_stencil_format,
		)?);
		Ok(())
	}

	/// Create a device-local buffer from a slice, initialized with `data` for `usage`.
	/// For stuff that isn't an array, just put the data into a single-element slice, like `[data]`.
	fn new_buffer<T>(&mut self, data: &[T], usage: BufferUsage) -> crate::Result<Subbuffer<[T]>>
	where
		T: BufferContents + Copy,
	{
		let data_len = data.len().try_into().unwrap();
		let data_size_bytes = std::mem::size_of_val(data);
		let buf;
		if self.direct_buffer_write {
			// When possible, upload directly to the new buffer memory.
			log::debug!("Allocating direct buffer of {data_size_bytes} bytes");
			let buf_info = BufferCreateInfo {
				usage,
				..Default::default()
			};
			let alloc_info = AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter {
					required_flags: MemoryPropertyFlags::HOST_VISIBLE,
					..MemoryTypeFilter::PREFER_DEVICE
				},
				..Default::default()
			};
			buf = Buffer::new_slice(self.memory_allocator.clone(), buf_info, alloc_info, data_len)?;
			buf.write().unwrap().copy_from_slice(data);
		} else {
			// If direct uploads aren't possible, create a staging buffer on the CPU side,
			// then submit a transfer command to the new buffer on the GPU side.
			log::debug!("Allocating buffer of {data_size_bytes} bytes");
			let buf_info = BufferCreateInfo {
				usage: usage | BufferUsage::TRANSFER_DST,
				..Default::default()
			};
			let alloc_info = AllocationCreateInfo::default();
			buf = Buffer::new_slice(self.memory_allocator.clone(), buf_info, alloc_info, data_len)?;
			self.transfer_manager.copy_to_buffer(data, buf.clone())?;
		}
		Ok(buf)
	}

	pub fn new_image<Px>(&mut self, data: &[Px], create_info: ImageCreateInfo) -> crate::Result<Arc<Image>>
	where
		Px: BufferContents + Copy,
	{
		let alloc_info = AllocationCreateInfo::default();
		let image = Image::new(self.memory_allocator.clone(), create_info, alloc_info)?;

		self.transfer_manager.copy_to_image(data, image.clone())?;

		Ok(image)
	}

	/// Load an image file as a 2D texture into memory.
	///
	/// The results of this are cached; if the image was already loaded, it'll use the loaded texture.
	pub fn new_texture(&mut self, path: &Path) -> crate::Result<Arc<ImageView>>
	{
		if let Some(tex) = self.textures.get(path) {
			return Ok(tex.clone());
		}

		// TODO: animated textures using APNG, animated JPEG-XL, or multi-layer DDS
		let (format, extent, mip_levels, img_raw) = load_texture(path)?;

		let image_info = ImageCreateInfo {
			format,
			extent: [extent[0], extent[1], 1],
			mip_levels,
			usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
			..Default::default()
		};
		let image = self.new_image(&img_raw, image_info)?;
		let view = ImageView::new_default(image)?;

		self.textures.insert(path.to_path_buf(), view.clone());

		Ok(view)
	}

	/// Get the color and depth images. They'll be resized before being returned if the swapchain
	/// image extent changed.
	fn get_images(&mut self) -> crate::Result<(Arc<ImageView>, Arc<ImageView>)>
	{
		let extent2 = self.swapchain.dimensions();
		let extent = [extent2[0], extent2[1], 1];
		if Some(extent) != self.color_image.as_ref().map(|view| view.image().extent()) {
			let color_create_info = ImageCreateInfo {
				format: Format::R16G16B16A16_SFLOAT,
				extent,
				usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
				..Default::default()
			};
			let color_image = Image::new(
				self.memory_allocator.clone(),
				color_create_info,
				AllocationCreateInfo::default(),
			)?;
			let color = ImageView::new_default(color_image)?;
			self.color_image = Some(color.clone());

			let depth_create_info = ImageCreateInfo {
				format: self.depth_stencil_format,
				extent,
				usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
				..Default::default()
			};
			let depth_image = Image::new(
				self.memory_allocator.clone(),
				depth_create_info,
				AllocationCreateInfo::default(),
			)?;
			let depth = ImageView::new_default(depth_image)?;
			self.depth_image = Some(depth);

			let set_layout = self.color_set_layout.clone();
			let set_write = [WriteDescriptorSet::image_view(0, color)];
			let set = PersistentDescriptorSet::new(&self.color_set_allocator, set_layout, set_write, [])?;
			self.color_set = Some(set);
		}

		Ok((self.color_image.clone().unwrap(), self.depth_image.clone().unwrap()))
	}

	fn present_image(&mut self, mut cb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> crate::Result<()>
	{
		if let Some((swapchain_image, acquire_future)) = self.swapchain.get_next_image()? {
			let color_image = self.color_image.as_ref().unwrap().image().clone();

			if self.swapchain.color_space() == ColorSpace::SrgbNonLinear {
				// perform gamma correction
				let image_extent = color_image.extent();
				let workgroups_x = image_extent[0].div_ceil(64);
				let layout = self.gamma_pipeline.layout().clone();
				cb.bind_pipeline_compute(self.gamma_pipeline.clone())?
					.bind_descriptor_sets(PipelineBindPoint::Compute, layout, 0, self.color_set.clone().unwrap())?
					.dispatch([workgroups_x, image_extent[1], 1])?;
			}

			cb.blit_image(BlitImageInfo::images(color_image, swapchain_image))?;

			let built_cb = cb.build()?;
			let transfer_future = self.transfer_manager.take_transfer_future();
			let graphics_queue = self.graphics_queue.clone();

			// wait for any transfers to finish
			// (ideally we'd use a semaphore, but it's borked in Vulkano right now)
			if let Some(f) = &transfer_future {
				f.wait(None)?;
			}

			// submit the built command buffer, presenting it if possible
			self.swapchain
				.present(graphics_queue, built_cb, acquire_future, transfer_future)?;
		}

		Ok(())
	}

	/// Check if the window has been resized since the last frame submission.
	pub fn window_resized(&self) -> bool
	{
		self.swapchain.extent_changed()
	}

	pub fn set_fullscreen(&self, fullscreen: bool)
	{
		self.swapchain.set_fullscreen(fullscreen)
	}
	pub fn is_fullscreen(&self) -> bool
	{
		self.swapchain.is_fullscreen()
	}

	pub fn reset_window_min_inner_size(&self)
	{
		self.swapchain.reset_min_inner_size()
	}
	pub fn swapchain_dimensions(&self) -> [u32; 2]
	{
		self.swapchain.dimensions()
	}

	/// Get the delta time for last frame.
	pub fn delta(&self) -> std::time::Duration
	{
		self.swapchain.delta()
	}
}

mod compute_gamma
{
	vulkano_shaders::shader! {
		ty: "compute",
		src: r"
			#version 460

			layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

			layout(binding = 0, rgba16f) uniform restrict image2D color_image;

			void main()
			{
				const float gamma = 1.0 / 2.2;

				ivec2 image_coord = ivec2(gl_GlobalInvocationID.xy);
				vec3 rgb_lin = imageLoad(color_image, image_coord).rgb;

				float r = pow(rgb_lin.r, gamma);
				float g = pow(rgb_lin.g, gamma);
				float b = pow(rgb_lin.b, gamma);

				imageStore(color_image, image_coord, vec4(r, g, b, 1.0));
			}
		",
	}
}

fn load_texture(path: &Path) -> crate::Result<(Format, [u32; 2], u32, Vec<u8>)>
{
	log::info!("Loading texture file '{}'...", path.display());
	match path.extension().and_then(|ext| ext.to_str()) {
		Some("dds") => {
			let dds_file = std::fs::File::open(path).map_err(|e| EngineError::new("couldn't open DDS file", e))?;
			let dds = ddsfile::Dds::read(dds_file).map_err(|e| EngineError::new("failed to read DDS file", e))?;

			// BC7_UNorm is treated as sRGB for now since Compressonator doesn't support converting to
			// BC7_UNorm_sRGB, even though the data itself appears to be in sRGB gamma.
			let vk_fmt = match dds.get_dxgi_format() {
				Some(DxgiFormat::BC1_UNorm_sRGB) => Format::BC1_RGBA_SRGB_BLOCK,
				Some(DxgiFormat::BC4_UNorm) => Format::BC4_UNORM_BLOCK,
				Some(DxgiFormat::BC5_UNorm) => Format::BC5_UNORM_BLOCK,
				Some(DxgiFormat::BC7_UNorm) => Format::BC7_SRGB_BLOCK,
				Some(DxgiFormat::BC7_UNorm_sRGB) => Format::BC7_SRGB_BLOCK,
				Some(format) => {
					let e = UnsupportedDdsFormat { format };
					return Err(EngineError::new("failed to read DDS file", e));
				}
				None => return Err("DDS file doesn't have a DXGI format".into()),
			};
			let dim = [dds.get_width(), dds.get_height()];
			let mip_count = dds.get_num_mipmap_levels();

			Ok((vk_fmt, dim, mip_count, dds.data))
		}
		_ => {
			// Load other formats such as PNG into an 8bpc sRGB RGBA image.
			let img = image::io::Reader::open(path)
				.map_err(|e| EngineError::new("failed to open image file", e))?
				.decode()
				.map_err(|e| EngineError::new("failed to decode image file", e))?
				.into_rgba8();
			Ok((Format::R8G8B8A8_SRGB, img.dimensions().into(), 1, img.into_raw()))
		}
	}
}

#[derive(Debug)]
struct UnsupportedDdsFormat
{
	format: DxgiFormat,
}
impl std::error::Error for UnsupportedDdsFormat
{
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)>
	{
		None
	}
}
impl std::fmt::Display for UnsupportedDdsFormat
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "DDS format '{:?}' is unsupported", self.format)
	}
}

/// Calculate the size (in bytes) that a mip level with the given format, width, and height would
/// take up.
///
/// This does not take array layers into account; the returned value should be multiplied by the
/// array layer count.
fn get_mip_size(format: Format, mip_width: u32, mip_height: u32) -> DeviceSize
{
	let block_extent = format.block_extent();
	let block_size = format.block_size();
	let x_blocks = mip_width.div_ceil(block_extent[0]) as DeviceSize;
	let y_blocks = mip_height.div_ceil(block_extent[1]) as DeviceSize;
	x_blocks * y_blocks * block_size
}

//
/* Render workload */
//
pub fn render_workload() -> Workload
{
	(submit_transfers, model::draw_workload, draw_ui, submit_frame).into_workload()
}

fn submit_transfers(mut render_ctx: UniqueViewMut<RenderContext>) -> crate::Result<()>
{
	render_ctx.transfer_manager.submit_transfers()
}
fn draw_ui(render_ctx: UniqueView<RenderContext>, mut canvas: UniqueViewMut<Canvas>) -> crate::Result<()>
{
	canvas.draw(&render_ctx)
}

// Submit all the command buffers for this frame to actually render them to the image.
fn submit_frame(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut mesh_manager: UniqueViewMut<MeshManager>,
	mut canvas: UniqueViewMut<Canvas>,
	mut light_manager: UniqueViewMut<LightManager>,
	camera_manager: UniqueView<CameraManager>,
) -> crate::Result<()>
{
	// If the window is minimized, don't present an image. This must be done because the window
	// (often on Windows) may report an inner width or height of 0, which we can't resize the
	// swapchain to. Presenting swapchain images anyways would cause an "out of date" error too.
	if render_ctx.swapchain.is_minimized() {
		return Ok(());
	}

	let (color_image, depth_image) = render_ctx.get_images()?;
	let mut primary_cb_builder = AutoCommandBufferBuilder::primary(
		&render_ctx.command_buffer_allocator,
		render_ctx.graphics_queue.queue_family_index(),
		CommandBufferUsage::OneTimeSubmit,
	)?;

	// shadows
	light_manager.execute_shadow_rendering(&mut primary_cb_builder)?;

	// skybox (effectively clears the image)
	if let Some(skybox) = &render_ctx.skybox {
		skybox.draw(&mut primary_cb_builder, color_image.clone(), camera_manager.sky_projview())?;
	}

	// 3D
	mesh_manager.execute_rendering(&mut primary_cb_builder, color_image.clone(), depth_image.clone())?;

	// 3D OIT
	let memory_allocator = render_ctx.memory_allocator.clone();
	if let Some(transparency_renderer) = &mut render_ctx.transparency_renderer {
		transparency_renderer.process_transparency(
			&mut primary_cb_builder,
			color_image.clone(),
			depth_image,
			memory_allocator,
		)?;
	}

	// UI
	canvas.execute_rendering(&mut primary_cb_builder, color_image)?;

	render_ctx.present_image(primary_cb_builder)
}

//
/* Vulkan initialization */
//
fn get_physical_device(app_name: &str, event_loop: &EventLoop<()>) -> crate::Result<Arc<PhysicalDevice>>
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
	let inst_create_info = InstanceCreateInfo {
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
	log::info!("Enabling instance extensions: {:?}", &inst_create_info.enabled_extensions);

	let physical_devices: smallvec::SmallVec<[_; 4]> = Instance::new(lib, inst_create_info)
		.map_err(|e| EngineError::new("failed to create Vulkan instance", e.unwrap()))?
		.enumerate_physical_devices()
		.map_err(|e| EngineError::new("failed to enumerate physical devices", e))?
		.collect();

	log::info!("Available Vulkan physical devices:");
	for (i, pd) in physical_devices.iter().enumerate() {
		let device_name = &pd.properties().device_name;
		let driver_name = pd.properties().driver_name.as_ref().map_or("[no driver name]", |name| name);
		let driver_info = pd.properties().driver_info.as_ref().map_or("[no driver info]", |name| name);
		log::info!("{i}: {device_name} (driver: {driver_name}, {driver_info})");
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

/// Create the Vulkan logical device. Returns a graphics queue (which owns the device) and an
/// optional transfer queue.
fn vulkan_setup(app_name: &str, event_loop: &EventLoop<()>) -> crate::Result<(Arc<Queue>, Option<Arc<Queue>>)>
{
	let physical_device = get_physical_device(app_name, event_loop)?;

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

fn check_direct_buffer_write(physical_device: &Arc<PhysicalDevice>) -> bool
{
	match physical_device.properties().device_type {
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
	}
}
