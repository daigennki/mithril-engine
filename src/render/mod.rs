/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod swapchain;
pub mod pipeline;
pub mod texture;
pub mod command_buffer;
pub mod model;
pub mod skybox;

use std::sync::Arc;
use std::collections::{ LinkedList, HashMap };
use std::fmt::Debug;
use std::path::{ Path, PathBuf };
use vulkano_win::VkSurfaceBuild;
use winit::window::WindowBuilder;
use vulkano::device::{ Queue, QueueFamilyProperties, physical::{ PhysicalDeviceType, PhysicalDevice } };
use vulkano::command_buffer::{ PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer, CopyBufferInfo, CopyBufferToImageInfo };
use vulkano::descriptor_set::{ WriteDescriptorSet, PersistentDescriptorSet };
use vulkano::format::Format;
use vulkano::buffer::{ 
	DeviceLocalBuffer, BufferUsage, cpu_access::CpuAccessibleBuffer, TypedBufferAccess
};
use vulkano::render_pass::Framebuffer;
use vulkano::memory::DeviceMemoryError;
use vulkano::sync::FlushError;
use vulkano::image::{ ImageDimensions, MipmapsCount };
use vulkano::pipeline::graphics::viewport::Viewport;

use command_buffer::CommandBuffer;
use model::Model;
use crate::GenericEngineError;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	swapchain: swapchain::Swapchain,
	dev_queue: Arc<Queue>,	// this also owns the logical device

	staging_work_queue: LinkedList<StagingWork>,

	// Loaded 3D models, with the key being the path relative to the current working directory.
	models: HashMap<PathBuf, Arc<Model>>,

	// User-accessible material pipelines; these will have their viewports resized
	// when the window size changes
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
			//.with_resizable(false)
			.build_vk_surface(&event_loop, dev_queue.device().instance().clone())?;

		Ok(RenderContext{
			swapchain: swapchain::Swapchain::new(dev_queue.device().clone(), window_surface)?,
			dev_queue: dev_queue.clone(),
			staging_work_queue: LinkedList::new(),
			models: HashMap::new(),
			material_pipelines: HashMap::new()
		})
	}

	/// Load a material shader pipeline into memory.
	/// The definition file name must be in the format "[name].yaml" and stored in the "shaders" folder.
	pub fn load_material_pipeline(&mut self, filename: &str) -> Result<(), GenericEngineError>
	{
		let name = filename.split_once('.')
			.ok_or(format!("Invalid material pipeline definition file name '{}'", filename))?
			.0.to_string();
		self.material_pipelines.insert(name, pipeline::Pipeline::new_from_yaml(filename, self.swapchain.render_pass())?);
		Ok(())
	}

	/// Get a 3D model from `path`, relative to the current working directory. 
	/// This attempts loading if it hasn't been loaded into memory yet.
	/// `use_embedded_materials` only takes effect if the model hasn't been loaded yet.
	pub fn get_model(&mut self, path: &Path, use_embedded_materials: bool) -> Result<Arc<Model>, GenericEngineError>
	{
		Ok(/*match self.models.get(path) {
			Some(model) => {
				log::info!("Reusing loaded model '{}'", path.display());
				model.clone()
			},
			None =>*/ {
				let new_model = Arc::new(Model::new(self, path, use_embedded_materials)?);
				self.models.insert(path.to_path_buf(), new_model.clone());
				new_model
			}
		/*}*/)
	}

	pub fn new_texture(&mut self, path: &Path) -> Result<texture::Texture, GenericEngineError>
	{
		let (tex, staging_work) = texture::Texture::new(self.dev_queue.clone(), path)?;
		self.staging_work_queue.push_back(staging_work.into());
		Ok(tex)
	}
	
	pub fn new_cubemap_texture(&mut self, faces: [PathBuf; 6])
		-> Result<texture::CubemapTexture, GenericEngineError>
	{
		let (tex, staging_work) = texture::CubemapTexture::new(self.dev_queue.clone(), faces)?;
		self.staging_work_queue.push_back(staging_work.into());
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
		let (tex, staging_work) = texture::Texture::new_from_iter(
			self.dev_queue.clone(), 
			iter,
			vk_fmt, 
			dimensions, 
			mip,
		)?;
		self.staging_work_queue.push_back(staging_work.into());
		Ok(tex)
	}

	/// Create an immutable buffer, initialized with `data` for `usage`.
	pub fn new_buffer_from_iter<I,T>(&mut self, data: I, mut usage: BufferUsage) 
		-> Result<Arc<DeviceLocalBuffer<[T]>>, GenericEngineError>
		where
			I: IntoIterator<Item = T>,
			I::IntoIter: ExactSizeIterator,
			[T]: vulkano::buffer::BufferContents, 
	{
		let staging_usage = BufferUsage{ transfer_src: true, ..BufferUsage::empty() };
		let staging_buf = CpuAccessibleBuffer::from_iter(self.dev_queue.device().clone(), staging_usage, false, data)?;
		usage.transfer_dst = true;
		let qfi = [ self.dev_queue.queue_family_index() ];
		let buf = DeviceLocalBuffer::array(self.dev_queue.device().clone(), staging_buf.len(), usage, qfi)?;
		self.staging_work_queue.push_back(CopyBufferInfo::buffers(staging_buf, buf.clone()).into());
		Ok(buf)
	}

	/// Create an immutable buffer, initialized with `data` for `usage`.
	pub fn new_buffer_from_data<T>(&mut self, data: T, mut usage: BufferUsage) 
		-> Result<Arc<DeviceLocalBuffer<T>>, GenericEngineError>
		where T: vulkano::buffer::BufferContents, 
	{
		let staging_usage = BufferUsage{ transfer_src: true, ..BufferUsage::empty() };
		let staging_buf = CpuAccessibleBuffer::from_data(self.dev_queue.device().clone(), staging_usage, false, data)?;
		usage.transfer_dst = true;
		let qfi = [ self.dev_queue.queue_family_index() ];
		let buf = DeviceLocalBuffer::new(self.dev_queue.device().clone(), usage, qfi)?;
		self.staging_work_queue.push_back(CopyBufferInfo::buffers(staging_buf, buf.clone()).into());
		Ok(buf)
	}

	/// Create a new CPU-accessible buffer, initialized with `data` for `usage`.
	pub fn new_cpu_buffer_from_iter<I, T>(&self, data: I, usage: BufferUsage)
		-> Result<Arc<CpuAccessibleBuffer<[T]>>, DeviceMemoryError>
		where
			I: IntoIterator<Item = T>,
			I::IntoIter: ExactSizeIterator,
			[T]: vulkano::buffer::BufferContents
	{
		CpuAccessibleBuffer::from_iter(self.dev_queue.device().clone(), usage, false, data)
	}

	/// Create a new CPU-accessible buffer, initialized with `data` for `usage`.
	pub fn new_cpu_buffer_from_data<T>(&self, data: T, usage: BufferUsage)
		-> Result<Arc<CpuAccessibleBuffer<T>>, DeviceMemoryError>
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
		CommandBuffer::<SecondaryAutoCommandBuffer>::new(self.dev_queue.clone(), Some(framebuffer))
	}

	/// Tell the swapchain to go to the next image.
	/// The image size *may* change here.
	/// This must only be called once per frame, at the beginning of each frame before any render pass.
	///
	/// This returns the framebuffer for the image, and if the images were resized, the new dimensions.
	pub fn next_swapchain_image(&mut self) 
		-> Result<(Arc<vulkano::render_pass::Framebuffer>, Option<[u32; 2]>), GenericEngineError>
	{
		let (next_img_fb, dimensions_changed) = self.swapchain.get_next_image()?;
		let new_dim = if dimensions_changed {
			Some(self.swapchain_dimensions())
		} else {
			None
		};

		Ok((next_img_fb, new_dim))
	}

	/// Build and take the command buffer for staging buffers and images.
	/// This will return `None` if there is nothing queued for staging.
	pub fn take_staging_command_buffer(&mut self) -> Result<Option<SecondaryAutoCommandBuffer>, GenericEngineError>
	{
		if self.staging_work_queue.is_empty() {
			return Ok(None)
		}
		let mut staging_cb = CommandBuffer::<SecondaryAutoCommandBuffer>::new(self.dev_queue.clone(), None)?;
		let work_count = self.staging_work_queue.len();
		log::debug!("Building a staging command buffer with {} copies", work_count);
		for work in std::mem::take(&mut self.staging_work_queue) {
			match work {
				StagingWork::CopyBuffer(info) => staging_cb.copy_buffer(info)?,
				StagingWork::CopyBufferToImage(info) => staging_cb.copy_buffer_to_image(info)?
			}
		}
		Ok(Some(staging_cb.build()?))
	}

	pub fn submit_commands(&mut self, built_cb: PrimaryAutoCommandBuffer) -> Result<(), GenericEngineError>
	{
		self.swapchain.submit_commands(built_cb, self.dev_queue.clone())
	}

	pub fn wait_for_fence(&self) -> Result<(), FlushError>
	{
		self.swapchain.wait_for_fence()
	}

	pub fn get_pipeline(&mut self, name: &str) -> Result<&pipeline::Pipeline, PipelineNotLoaded>
	{
		Ok(self.material_pipelines.get(name).ok_or(PipelineNotLoaded)?)
	}
	
	pub fn swapchain_dimensions(&self) -> [u32; 2]
	{
		self.swapchain.dimensions()
	}

	pub fn get_swapchain_viewport(&self) -> Viewport
	{
		self.swapchain.get_viewport()
	}

	/// Get the current swapchain framebuffer.
	pub fn get_current_framebuffer(&self) -> Arc<Framebuffer>
	{
		self.swapchain.get_current_framebuffer()
	}

	pub fn get_queue(&self) -> Arc<Queue>
	{
		self.dev_queue.clone()
	}
	pub fn get_surface(&self) -> Arc<vulkano::swapchain::Surface<winit::window::Window>>
	{
		self.swapchain.get_surface()
	}
}

#[derive(Debug)]
pub struct PipelineNotLoaded;
impl std::error::Error for PipelineNotLoaded {}
impl std::fmt::Display for PipelineNotLoaded {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "the specified pipeline is not loaded")
    }
}

enum StagingWork
{
	CopyBuffer(CopyBufferInfo),
	CopyBufferToImage(CopyBufferToImageInfo)
}
impl From<CopyBufferInfo> for StagingWork
{
	fn from(info: CopyBufferInfo) -> StagingWork
	{
		Self::CopyBuffer(info)
	}
}
impl From<CopyBufferToImageInfo> for StagingWork
{
	fn from(info: CopyBufferToImageInfo) -> StagingWork
	{
		Self::CopyBufferToImage(info)
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
	-> Result<(), vulkano::VulkanError>
{
	log::info!("Available Vulkan physical devices:");
	for (i, pd) in vkinst.enumerate_physical_devices()?.enumerate() {
		let driver_ver = decode_driver_version(pd.properties().driver_version, pd.properties().vendor_id);
		
		log::info!("{}: {} ({:?}), driver '{}' version {}.{}.{}.{} (Vulkan {})", 
			i, 
			pd.properties().device_name, 
			pd.properties().device_type,
			pd.properties().driver_name.clone().unwrap_or("unknown driver".into()), 
			driver_ver.0, driver_ver.1, driver_ver.2, driver_ver.3,
			pd.properties().api_version
		);
	}
	Ok(())
}
fn print_queue_families<'a>(queue_families: &[QueueFamilyProperties])
{
	log::info!("Available physical device queue families:");
	for (id, qf) in queue_families.iter().enumerate() {
		log::info!("{}: {} queue(s), {:?}", id, qf.queue_count, qf.queue_flags);
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
	inst_create_info.max_api_version = Some(vulkano::Version::V1_2);
	
	Ok(vulkano::instance::Instance::new(lib, inst_create_info)?)
}

/// Get the most appropriate GPU, along with a graphics queue family.
fn get_physical_device(vkinst: Arc<vulkano::instance::Instance>) 
	-> Result<(Arc<PhysicalDevice>, usize), GenericEngineError>
{	
	print_physical_devices(&vkinst)?;
	let dgpu = vkinst.enumerate_physical_devices()?
		.find(|pd| pd.properties().device_type == PhysicalDeviceType::DiscreteGpu);
	let igpu = vkinst.enumerate_physical_devices()?
		.find(|pd| pd.properties().device_type == PhysicalDeviceType::IntegratedGpu);

	// Try to use a discrete GPU. If there is no discrete GPU, use an integrated GPU instead.
	let physical_device = dgpu.or(igpu).ok_or("No GPUs were found!")?;

	log::info!("Using physical device: {}", physical_device.properties().device_name);

	// get queue family that supports graphics
	print_queue_families(physical_device.queue_family_properties());
	let (q_fam, _) = physical_device.queue_family_properties()
		.iter()
		.enumerate()
		.find(|(_, q)| q.queue_flags.graphics)
		.ok_or("No graphics queue family found!")?;

	Ok((physical_device, q_fam))
}

/// Set up the Vulkan instance, physical device, logical device, and queue.
/// The `Queue` this returns will own all of them.
fn vulkan_setup(game_name: &str) 
	-> Result<Arc<Queue>, GenericEngineError>
{
	let vkinst = create_vulkan_instance(game_name)?;
	let (physical_device, queue_family) = get_physical_device(vkinst.clone())?;

	// Select features and extensions.
	// The ones chosen here are practically universally supported by any device with Vulkan support.
	let dev_features = vulkano::device::Features{
		image_cube_array: true,
		independent_blend: true,
		sampler_anisotropy: true,
		texture_compression_bc: true,	// change this to ASTC or ETC2 if we want to support mobile platforms
		geometry_shader: true,
		..vulkano::device::Features::empty()
	};
	let dev_extensions = vulkano::device::DeviceExtensions{
		khr_swapchain: true,
		..vulkano::device::DeviceExtensions::empty()
	};
	
	let dev_create_info = vulkano::device::DeviceCreateInfo{
		enabled_extensions: dev_extensions,
		enabled_features: dev_features,
		queue_create_infos: vec![ vulkano::device::QueueCreateInfo{ queue_family_index: queue_family.try_into()?, ..Default::default() } ],
		..Default::default()
	};

	let (_, mut queues) = vulkano::device::Device::new(physical_device, dev_create_info)?;

	Ok(queues.next().ok_or("`vulkano::device::Device::new(...) returned 0 queues`")?)
}

