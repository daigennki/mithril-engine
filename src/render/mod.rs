/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod swapchain;
pub mod pipeline;
pub mod texture;
pub mod command_buffer;

use std::sync::Arc;
use std::collections::HashMap;
use std::fmt::Debug;
use vulkano_win::VkSurfaceBuild;
use winit::window::WindowBuilder;
use vulkano::device::{ Queue, physical::{ PhysicalDeviceType, PhysicalDevice, QueueFamily } };
use vulkano::command_buffer::{ PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer };
use vulkano::descriptor_set::{ WriteDescriptorSet, PersistentDescriptorSet };
use vulkano::format::Format;
use vulkano::buffer::{ 
	ImmutableBuffer, BufferUsage, cpu_access::CpuAccessibleBuffer, immutable::ImmutableBufferCreationError
};
use vulkano::render_pass::Framebuffer;
use vulkano::memory::DeviceMemoryAllocationError;
use vulkano::sync::{ GpuFuture };
use vulkano::image::{ ImageDimensions, MipmapsCount };

use command_buffer::CommandBuffer;
use crate::GenericEngineError;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	swapchain: swapchain::Swapchain,
	dev_queue: Arc<vulkano::device::Queue>,	// this also owns the logical device

	upload_futures: Option<Box<dyn vulkano::sync::GpuFuture + Send + Sync>>,
	upload_futures_count: usize,

	// User-accessible material pipelines; these will have their viewports resized
	// when the window size changes
	// TODO: give ownership of these to "Material" objects?
	material_pipelines: HashMap<String, pipeline::Pipeline>,
	
	// TODO: put non-material shaders (shadow filtering, post processing) into different containers
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) -> Result<Self, GenericEngineError>
	{
		let dev_queue = vulkan_setup(game_name)?;

		// create window
		let window_surface = WindowBuilder::new()
			.with_min_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
			.with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))	// TODO: load this from config
			.with_title(game_name)
			.with_resizable(false)
			.build_vk_surface(&event_loop, dev_queue.device().instance().clone())?;

		Ok(RenderContext{
			swapchain: swapchain::Swapchain::new(dev_queue.device().clone(), window_surface)?,
			dev_queue: dev_queue.clone(),
			upload_futures: None,
			upload_futures_count: 0,
			material_pipelines: HashMap::new()
		})
	}

	/// Load a material shader pipeline into memory.
	/// The definition file name must be in the format "[name].yaml" and stored in the "shaders" folder.
	pub fn load_material_pipeline(&mut self, definition_file: &str) -> Result<(), GenericEngineError>
	{
		let name = definition_file.split_once('.')
			.ok_or(format!("Invalid material pipeline definition file name '{}'", definition_file))?
			.0.to_string();
		let dim = self.swapchain.dimensions();
		self.material_pipelines.insert(
			name, 
			pipeline::Pipeline::new_from_yaml(definition_file, self.swapchain.render_pass(), dim[0], dim[1])?
		);
		Ok(())
	}
	
	fn join_future<F>(&mut self, next_future: F)
		where F: vulkano::sync::GpuFuture + 'static + Send + Sync
	{
		self.upload_futures = Some(
			match self.upload_futures.take() {
				Some(f) => Box::new(f.join(next_future)),
				None => Box::new(next_future)
			}
		);
		self.upload_futures_count += 1;
	}

	pub fn new_texture(&mut self, path: &std::path::Path) -> Result<texture::Texture, GenericEngineError>
	{
		let (tex, upload_future) = texture::Texture::new(self.dev_queue.clone(), path)?;
		self.join_future(upload_future);
		Ok(tex)
	}

	pub fn new_texture_from_iter<Px, I>(&mut self,	
		iter: I, 
		vk_fmt: Format, 
		dimensions: ImageDimensions,
		mip: MipmapsCount
	) 
	-> Result<texture::Texture, GenericEngineError>
	where
		[Px]: vulkano::buffer::BufferContents,
		I: IntoIterator<Item = Px>,
		I::IntoIter: ExactSizeIterator
	{
		let (tex, upload_future) = texture::Texture::new_from_iter(
			self.dev_queue.clone(), 
			iter,
			vk_fmt, 
			dimensions, 
			mip
		)?;
		self.join_future(upload_future);
		Ok(tex)
	}

	/// Create an immutable buffer, initialized with `data` for `usage`.
	pub fn new_buffer_from_iter<I,T>(&mut self, data: I, usage: BufferUsage) 
		-> Result<Arc<ImmutableBuffer<[T]>>, ImmutableBufferCreationError>
		where
			I: IntoIterator<Item = T>,
			I::IntoIter: ExactSizeIterator,
			[T]: vulkano::buffer::BufferContents, 
	{
		let (buf, upload_future) = ImmutableBuffer::from_iter(data, usage, self.dev_queue.clone())?;
		self.join_future(upload_future);
		Ok(buf)
	}

	/// Create an immutable buffer, initialized with `data` for `usage`.
	pub fn new_buffer_from_data<T>(&mut self, data: T, usage: BufferUsage) 
		-> Result<Arc<ImmutableBuffer<T>>, ImmutableBufferCreationError>
		where T: vulkano::buffer::BufferContents, 
	{
		let (buf, upload_future) = ImmutableBuffer::from_data(data, usage, self.dev_queue.clone())?;
		self.join_future(upload_future);
		Ok(buf)
	}

	/// Create a new CPU-accessible buffer, initialized with `data` for `usage`.
	pub fn new_cpu_buffer_from_iter<I, T>(&self, data: I, usage: BufferUsage)
		-> Result<Arc<CpuAccessibleBuffer<[T]>>, DeviceMemoryAllocationError>
		where
			I: IntoIterator<Item = T>,
			I::IntoIter: ExactSizeIterator,
			[T]: vulkano::buffer::BufferContents
	{
		CpuAccessibleBuffer::from_iter(self.dev_queue.device().clone(), usage, false, data)
	}

	/// Create a new CPU-accessible buffer, initialized with `data` for `usage`.
	pub fn new_cpu_buffer_from_data<T>(&self, data: T, usage: BufferUsage)
		-> Result<Arc<CpuAccessibleBuffer<T>>, DeviceMemoryAllocationError>
		where T: vulkano::buffer::BufferContents
	{
		CpuAccessibleBuffer::from_data(self.dev_queue.device().clone(), usage, false, data)
	}

	pub fn new_descriptor_set(&self, pipeline_name: &str, set: usize, writes: impl IntoIterator<Item = WriteDescriptorSet>)
		-> Result<Arc<PersistentDescriptorSet>, GenericEngineError>
	{
		self.material_pipelines.get(pipeline_name).ok_or(PipelineNotLoaded)?.new_descriptor_set(set, writes)
	}

	/// Issue a new primary command buffer builder to begin recording to.
	pub fn new_primary_command_buffer(&mut self) 
		-> Result<CommandBuffer<PrimaryAutoCommandBuffer>, GenericEngineError>
	{
		CommandBuffer::<PrimaryAutoCommandBuffer>::new(self.dev_queue.clone())
	}

	/// Issue a new secondary command buffer builder to begin recording to.
	/// It will be set up for drawing to `framebuffer`.
	pub fn new_secondary_command_buffer(&mut self, framebuffer: Arc<vulkano::render_pass::Framebuffer>) 
		-> Result<CommandBuffer<SecondaryAutoCommandBuffer>, GenericEngineError>
	{
		CommandBuffer::<SecondaryAutoCommandBuffer>::new(self.dev_queue.clone(), framebuffer)
	}

	/// Tell the swapchain to go to the next image.
	/// The image size may change here, in which case pipelines will be re-created with resized viewports.
	/// This must only be called once per frame, at the beginning of each frame before any render pass.
	///
	/// This returns the framebuffer for the image.
	pub fn next_swapchain_image(&mut self) -> Result<Arc<vulkano::render_pass::Framebuffer>, GenericEngineError>
	{
		let (next_img_fb, resize_viewports) = self.swapchain.get_next_image()?;

		if resize_viewports {
			let new_dimensions = self.swapchain.dimensions();
			log::debug!("Recreating pipelines with new viewport...");
			for (_, pl) in &mut self.material_pipelines {
				pl.resize_viewport(new_dimensions[0], new_dimensions[1])?;
			}
		}

		Ok(next_img_fb)
	}

	pub fn submit_commands(&mut self, built_cb: PrimaryAutoCommandBuffer) -> Result<(), GenericEngineError>
	{
		// consume the futures to join them upon submission
		if self.upload_futures_count > 0 {
			log::debug!("Joining a future of {} futures.", self.upload_futures_count);
		}
		self.upload_futures_count = 0;
		self.swapchain.submit_commands(built_cb, self.dev_queue.clone(), self.upload_futures.take())
	}

	pub fn get_pipeline(&mut self, name: &str) -> Result<&pipeline::Pipeline, PipelineNotLoaded>
	{
		Ok(self.material_pipelines.get(name).ok_or(PipelineNotLoaded)?)
	}
	
	pub fn swapchain_dimensions(&self) -> [u32; 2]
	{
		self.swapchain.dimensions()
	}

	/// Get the current swapchain framebuffer.
	pub fn get_current_framebuffer(&self) -> Arc<Framebuffer>
	{
		self.swapchain.get_current_framebuffer()
	}

	/*pub fn wait_for_fence(&self) -> Result<(), FlushError>
	{
		self.swapchain.wait_for_fence()
	}*/
}

#[derive(Debug)]
pub struct PipelineNotLoaded;
impl std::error::Error for PipelineNotLoaded {}
impl std::fmt::Display for PipelineNotLoaded {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "the specified pipeline is not loaded")
    }
}

fn decode_driver_version(version: u32, vendor_id: u32) -> (u32, u32, u32, u32)
{
	// NVIDIA
	if vendor_id == 4318 {
		return ((version >> 22) & 0x3ff,
		(version >> 14) & 0x0ff,
		(version >> 6) & 0x0ff,
		version & 0x003f)
	}
	
	// Intel (Windows only)
	#[cfg(target_family = "windows")]
	if vendor_id == 0x8086 {
		return ((version >> 14),
		version & 0x3fff,
		0, 0)
	}

	// others (use Vulkan version convention)
	((version >> 22),
	(version >> 12) & 0x3ff,
	version & 0xfff, 0)
}
fn print_physical_devices(vkinst: &Arc<vulkano::instance::Instance>)
{
	log::info!("Available Vulkan physical devices:");
	for pd in PhysicalDevice::enumerate(vkinst) {
		let driver_ver = decode_driver_version(pd.properties().driver_version, pd.properties().vendor_id);
		
		log::info!("{}: {} ({:?}), driver '{}' version {}.{}.{}.{} (Vulkan {})", 
			pd.index(), 
			pd.properties().device_name, 
			pd.properties().device_type,
			pd.properties().driver_name.clone().unwrap_or("unknown driver".into()), 
			driver_ver.0, driver_ver.1, driver_ver.2, driver_ver.3,
			pd.properties().api_version
		);
	}
}
fn print_queue_families<'a>(queue_families: impl ExactSizeIterator<Item = QueueFamily<'a>>)
{
	log::info!("Available physical device queue families:");
	for qf in queue_families {
		let qf_type = if qf.supports_graphics() {
			"graphics"
		} else if qf.supports_compute() {
			"compute"
		} else {
			"unknown"
		};
		let explicit_transfer = if qf.explicitly_supports_transfers() {
			"explicit"
		} else {
			"implicit"
		};
		log::info!("{}: {}, {} transfer support, {} queue(s)", qf.id(), qf_type, explicit_transfer, qf.queues_count());
	}
}

fn create_vulkan_instance(game_name: &str) -> Result<Arc<vulkano::instance::Instance>, GenericEngineError>
{
	// we'll need to enable the `enumerate_portability` extension if we want to use devices with non-conformant Vulkan
	// implementations like MoltenVK. for now, we can go without it.
	let vk_ext = vulkano_win::required_extensions();
	
	// only use the validation layer in debug builds
	#[cfg(debug_assertions)]
	let vk_layers = vec!["VK_LAYER_KHRONOS_validation".to_string()];
	#[cfg(not(debug_assertions))]
	let vk_layers: Vec<String> = vec![];

	let mut inst_create_info = vulkano::instance::InstanceCreateInfo::application_from_cargo_toml();
	inst_create_info.application_name = Some(game_name.to_string());
	inst_create_info.engine_name = Some("MithrilEngine".to_string());
	inst_create_info.engine_version = inst_create_info.application_version.clone();
	inst_create_info.enabled_extensions = vk_ext;
	inst_create_info.enabled_layers = vk_layers;
	inst_create_info.max_api_version = Some(vulkano::Version::V1_2);
	
	Ok(vulkano::instance::Instance::new(inst_create_info)?)
}

/// Get the most appropriate GPU, along with a graphics queue family.
fn get_physical_device<'a>(vkinst: &'a Arc<vulkano::instance::Instance>) 
	-> Result<(PhysicalDevice<'a>, QueueFamily<'a>), GenericEngineError>
{	
	print_physical_devices(&vkinst);

	let physical_device = PhysicalDevice::enumerate(&vkinst)
		.find(|pd| pd.properties().device_type == PhysicalDeviceType::DiscreteGpu)	// Look for a discrete GPU.
		.or_else(|| {
			// If there is no discrete GPU, try to look for an integrated GPU instead.
			PhysicalDevice::enumerate(&vkinst)
				.find(|pd| pd.properties().device_type == PhysicalDeviceType::IntegratedGpu)	
		}).ok_or("No GPUs were found!")?;

	log::info!("Using physical device: {}", physical_device.properties().device_name);

	// get queue family that supports graphics
	print_queue_families(physical_device.queue_families());
	let q_fam = physical_device.queue_families().find(|q| q.supports_graphics())
			.ok_or("No graphics queue family found!")?;

	Ok((physical_device, q_fam))
}

/// Set up the Vulkan instance, physical device, logical device, and queue.
/// The `Queue` this returns will own all of them.
fn vulkan_setup(game_name: &str) 
	-> Result<Arc<Queue>, GenericEngineError>
{
	let vkinst = create_vulkan_instance(game_name)?;
	let (physical_device, queue_family) = get_physical_device(&vkinst)?;

	log::info!("Dumping physical device properties:");
	log::info!("{:#?}", physical_device.properties());

	// Select features and extensions.
	// The ones chosen here are practically universally supported by any device with Vulkan support.
	let dev_features = vulkano::device::Features{
		image_cube_array: true,
		independent_blend: true,
		sampler_anisotropy: true,
		texture_compression_bc: true,	// change this to ASTC or ETC2 if we want to support mobile platforms
		geometry_shader: true,
		..vulkano::device::Features::none()
	};
	let dev_extensions = vulkano::device::DeviceExtensions{
		khr_swapchain: true,
		..vulkano::device::DeviceExtensions::none()
	};
	
	let dev_create_info = vulkano::device::DeviceCreateInfo{
		enabled_extensions: dev_extensions,
		enabled_features: dev_features,
		queue_create_infos: vec![ vulkano::device::QueueCreateInfo::family(queue_family) ],
		..Default::default()
	};

	let (_, mut queues) = vulkano::device::Device::new(physical_device, dev_create_info)?;

	Ok(queues.next().ok_or("`vulkano::device::Device::new(...) returned 0 queues`")?)
}

