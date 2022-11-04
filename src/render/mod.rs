/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod model;
pub mod pipeline;
pub mod skybox;
mod swapchain;
pub mod texture;

use std::collections::{HashMap, LinkedList};
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer, TypedBufferAccess};
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, CommandBufferBeginError, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderPassInfo,
	CommandBufferInheritanceRenderPassType, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo, PipelineExecutionError,
	PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::{DescriptorSetsCollection, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{
	physical::{PhysicalDevice, PhysicalDeviceType},
	Queue, QueueCreateInfo, QueueFamilyProperties,
};
use vulkano::format::{ClearValue, Format};
use vulkano::image::{ImageDimensions, MipmapsCount, ImageUsage, AttachmentImage, view::ImageView, ImageViewAbstract};
use vulkano::memory::DeviceMemoryError;
use vulkano::pipeline::{graphics::viewport::Viewport, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass, FramebufferCreateInfo};
use vulkano_win::VkSurfaceBuild;
use winit::window::WindowBuilder;

use crate::GenericEngineError;
use model::Model;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	swapchain: swapchain::Swapchain,
	graphics_queue: Arc<Queue>,         // this also owns the logical device
	transfer_queue: Option<Arc<Queue>>, // if there is a dedicated transfer queue, use it for transfers

	staging_work: LinkedList<StagingWork>,

	// Loaded 3D models, with the key being the path relative to the current working directory.
	models: HashMap<PathBuf, Arc<Model>>,

	// User-accessible material pipelines; these will have their viewports resized
	// when the window size changes
	material_pipelines: HashMap<String, pipeline::Pipeline>,

	
	// TODO: put non-material shaders (shadow filtering, post processing) into different containers
	last_frame_presented: std::time::Instant,
	frame_time: std::time::Duration,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) -> Result<Self, GenericEngineError>
	{
		let (graphics_queue, transfer_queue) = vulkan_setup(game_name)?;

		// create window
		let window_surface = WindowBuilder::new()
			.with_min_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
			.with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)) // TODO: load this from config
			.with_title(game_name)
			//.with_resizable(false)
			.build_vk_surface(&event_loop, graphics_queue.device().instance().clone())?;

		let swapchain = swapchain::Swapchain::new(graphics_queue.device().clone(), window_surface)?;

		Ok(RenderContext {
			swapchain,
			graphics_queue,
			transfer_queue,
			staging_work: LinkedList::new(),
			models: HashMap::new(),
			material_pipelines: HashMap::new(),
			last_frame_presented: std::time::Instant::now(),
			frame_time: std::time::Duration::ZERO,
		})
	}

	/// Load a material shader pipeline into memory.
	/// The definition file name must be in the format "[name].yaml" and stored in the "shaders" folder.
	pub fn load_material_pipeline(&mut self, filename: &str) -> Result<(), GenericEngineError>
	{
		let name = filename
			.split_once('.')
			.ok_or(format!("Invalid material pipeline definition file name '{}'", filename))?
			.0
			.to_string();
		self.material_pipelines
			.insert(name, pipeline::Pipeline::new_from_yaml(filename, self.swapchain.render_pass().first_subpass())?);
		Ok(())
	}
	pub fn load_mat_pipeline_manual(&mut self, name: &str, pipeline: pipeline::Pipeline)
	{
		self.material_pipelines.insert(name.to_string(), pipeline);
	}

	/// Get a 3D model from `path`, relative to the current working directory.
	/// This attempts loading if it hasn't been loaded into memory yet.
	/// `use_embedded_materials` only takes effect if the model hasn't been loaded yet.
	pub fn get_model(&mut self, path: &Path, use_embedded_materials: bool) -> Result<Arc<Model>, GenericEngineError>
	{
		Ok(
			/*match self.models.get(path) {
			Some(model) => {
				log::info!("Reusing loaded model '{}'", path.display());
				model.clone()
			},
			None =>*/
			{
				let new_model = Arc::new(Model::new(self, path, use_embedded_materials)?);
				self.models.insert(path.to_path_buf(), new_model.clone());
				new_model
			}, /*}*/
		)
	}

	pub fn new_texture(&mut self, path: &Path) -> Result<texture::Texture, GenericEngineError>
	{
		let (tex, staging_work) = texture::Texture::new(self.graphics_queue.device().clone(), path)?;
		self.staging_work.push_back(staging_work.into());
		Ok(tex)
	}

	pub fn new_cubemap_texture(&mut self, faces: [PathBuf; 6]) -> Result<texture::CubemapTexture, GenericEngineError>
	{
		let (tex, staging_work) = texture::CubemapTexture::new(self.graphics_queue.device().clone(), faces)?;
		self.staging_work.push_back(staging_work.into());
		Ok(tex)
	}

	pub fn new_texture_from_iter<Px, I>(
		&mut self, iter: I, vk_fmt: Format, dimensions: ImageDimensions, mip: MipmapsCount,
	) -> Result<texture::Texture, GenericEngineError>
	where
		[Px]: vulkano::buffer::BufferContents,
		I: IntoIterator<Item = Px>,
		I::IntoIter: ExactSizeIterator,
	{
		let (tex, staging_work) =
			texture::Texture::new_from_iter(self.graphics_queue.device().clone(), iter, vk_fmt, dimensions, mip)?;
		self.staging_work.push_back(staging_work.into());
		Ok(tex)
	}

	/// Create an immutable buffer, initialized with `data` for `usage`.
	pub fn new_buffer_from_iter<I, T>(
		&mut self, data: I, mut usage: BufferUsage,
	) -> Result<Arc<DeviceLocalBuffer<[T]>>, GenericEngineError>
	where
		I: IntoIterator<Item = T>,
		I::IntoIter: ExactSizeIterator,
		[T]: vulkano::buffer::BufferContents,
	{
		let staging_usage = BufferUsage { transfer_src: true, ..BufferUsage::empty() };
		let staging_buf = CpuAccessibleBuffer::from_iter(self.graphics_queue.device().clone(), staging_usage, false, data)?;
		usage.transfer_dst = true;
		let buf = DeviceLocalBuffer::array(
			self.graphics_queue.device().clone(),
			staging_buf.len(),
			usage,
			self.get_queue_families(),
		)?;
		self.staging_work
			.push_back(CopyBufferInfo::buffers(staging_buf, buf.clone()).into());
		Ok(buf)
	}

	/// Create an immutable buffer, initialized with `data` for `usage`.
	pub fn new_buffer_from_data<T>(
		&mut self, data: T, mut usage: BufferUsage,
	) -> Result<Arc<DeviceLocalBuffer<T>>, GenericEngineError>
	where
		T: vulkano::buffer::BufferContents,
	{
		let staging_usage = BufferUsage { transfer_src: true, ..BufferUsage::empty() };
		let staging_buf = CpuAccessibleBuffer::from_data(self.graphics_queue.device().clone(), staging_usage, false, data)?;
		usage.transfer_dst = true;
		let buf = DeviceLocalBuffer::new(self.graphics_queue.device().clone(), usage, self.get_queue_families())?;
		self.staging_work
			.push_back(CopyBufferInfo::buffers(staging_buf, buf.clone()).into());
		Ok(buf)
	}

	/*/// Create a new CPU-accessible buffer, initialized with `data` for `usage`.
	pub fn new_cpu_buffer_from_iter<I, T>(&self, data: I, usage: BufferUsage)
		-> Result<Arc<CpuAccessibleBuffer<[T]>>, DeviceMemoryError>
		where
			I: IntoIterator<Item = T>,
			I::IntoIter: ExactSizeIterator,
			[T]: vulkano::buffer::BufferContents
	{
		CpuAccessibleBuffer::from_iter(self.graphics_queue.device().clone(), usage, false, data)
	}*/

	/// Create a new pair of CPU-accessible buffer pool and device-local buffer, which will be initialized with `data` for `usage`.
	/// The CPU-accessible buffer pool is used for staging, from which data will be copied to the device-local buffer.
	pub fn new_cpu_buffer_from_data<T>(
		&mut self, data: T, mut usage: BufferUsage,
	) -> Result<(CpuBufferPool<T>, Arc<DeviceLocalBuffer<T>>), DeviceMemoryError>
	where
		[T]: vulkano::buffer::BufferContents,
		T: Send + Sync + bytemuck::Pod,
	{
		let cpu_buf = CpuBufferPool::upload(self.graphics_queue.device().clone());
		usage.transfer_dst = true;
		let gpu_buf = DeviceLocalBuffer::new(self.graphics_queue.device().clone(), usage, self.get_queue_families())?;
		self.staging_work
			.push_back(CopyBufferInfo::buffers(cpu_buf.from_data(data)?, gpu_buf.clone()).into());
		Ok((cpu_buf, gpu_buf))
	}

	fn get_queue_families(&self) -> Vec<u32>
	{
		self.graphics_queue
			.device()
			.active_queue_family_indices()
			.into()
	}

	/// Queue a buffer copy which will be executed before the next image submission.
	pub fn copy_buffer(&mut self, src: Arc<dyn BufferAccess>, dst: Arc<dyn BufferAccess>)
	{
		self.staging_work
			.push_back(CopyBufferInfo::buffers(src, dst).into())
	}

	pub fn new_descriptor_set(
		&self, pipeline_name: &str, set: usize, writes: impl IntoIterator<Item = WriteDescriptorSet>,
	) -> Result<Arc<PersistentDescriptorSet>, GenericEngineError>
	{
		self.material_pipelines
			.get(pipeline_name)
			.ok_or(PipelineNotLoaded)?
			.new_descriptor_set(set, writes)
	}

	/// Issue a new secondary command buffer builder to begin recording to.
	/// It will be set up for drawing to `framebuffer` in its first subpass, 
	/// and will have a command added to set its viewport to fill the extent of the framebuffer.
	pub fn new_secondary_command_buffer(
		&self, framebuffer: Arc<Framebuffer>,
	) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, CommandBufferBeginError>
	{
		let inherit_rp = CommandBufferInheritanceRenderPassType::BeginRenderPass(CommandBufferInheritanceRenderPassInfo {
			subpass: framebuffer.render_pass().clone().first_subpass(),
			framebuffer: Some(framebuffer.clone()),
		});
		let inheritance = CommandBufferInheritanceInfo {
			render_pass: Some(inherit_rp),
			..Default::default()
		};
		let mut cb = AutoCommandBufferBuilder::secondary(
			self.graphics_queue.device().clone(),
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
			inheritance,
		)?;

		// set viewport state
		let fb_extent = framebuffer.extent();
		let viewport = Viewport { 
			origin: [ 0.0, 0.0 ],
			dimensions: [ fb_extent[0] as f32, fb_extent[1] as f32 ],
			depth_range: 0.0..1.0
		};
		cb.set_viewport(0, [viewport]);

		Ok(cb)
	}

	/*/// Create a new single-pass framebuffer, automatically creating images for it to fit the given render pass, extent, and
	/// usage.
	pub fn new_single_pass_framebuffer_auto_images(
		&self, render_pass: Arc<RenderPass>, extent: [u32; 2], usage: ImageUsage
	) -> Result<Arc<Framebuffer>, GenericEngineError>
	{
		let attachments = render_pass
			.attachments()
			.iter()
			.map(|a| {
				let img_fmt = a.format.unwrap();
				let img = AttachmentImage::with_usage(self.graphics_queue.device().clone(), extent, img_fmt, usage)?;
				let view: Arc<dyn ImageViewAbstract> = ImageView::new_default(img)?;
				Ok(view)
			})
			.collect::<Result<_, GenericEngineError>>()?;
			
		let fb_create_info = FramebufferCreateInfo { attachments, extent, layers: 1, ..Default::default() };

		Ok(Framebuffer::new(render_pass, fb_create_info)?)
	}*/

	/// Tell the swapchain to go to the next image.
	/// The image size *may* change here.
	/// This must only be called once per frame, at the beginning of each frame before any render pass.
	///
	/// This returns the framebuffer for the image, and if the images were resized, the new dimensions.
	pub fn next_swapchain_image(
		&mut self,
	) -> Result<(Arc<Framebuffer>, Option<[u32; 2]>), GenericEngineError>
	{
		let (next_img_fb, dimensions_changed) = self.swapchain.get_next_image()?;
		let new_dim = dimensions_changed.then(|| self.swapchain_dimensions());

		Ok((next_img_fb, new_dim))
	}

	/// Build and take the command buffer for staging buffers and images, if anything needs to be copied.
	fn build_staging_cb(&mut self) -> Result<Option<PrimaryAutoCommandBuffer>, GenericEngineError>
	{
		// TODO: we might be able to build this command buffer in a separate thread, popping work out of
		// the `LinkedList` as soon as possible.
		if self.staging_work.is_empty() {
			Ok(None)
		} else {
			let cb_queue = self
				.transfer_queue
				.as_ref()
				.cloned()
				.unwrap_or_else(|| self.graphics_queue.clone());

			let mut staging_cb_builder = AutoCommandBufferBuilder::primary(
				cb_queue.device().clone(),
				cb_queue.queue_family_index(),
				CommandBufferUsage::OneTimeSubmit,
			)?;

			while let Some(work) = self.staging_work.pop_front() {
				match work {
					StagingWork::CopyBuffer(info) => staging_cb_builder.copy_buffer(info)?,
					StagingWork::CopyBufferToImage(info) => staging_cb_builder.copy_buffer_to_image(info)?,
				};
			}

			Ok(Some(staging_cb_builder.build()?))
		}
	}

	fn submit_commands(&mut self, built_cb: PrimaryAutoCommandBuffer) -> Result<(), GenericEngineError>
	{
		let staging_cb = self.build_staging_cb()?;

		self.swapchain.submit_commands(
			built_cb,
			self.graphics_queue.clone(),
			staging_cb,
			self.transfer_queue.as_ref().cloned(),
		)?;

		let now = std::time::Instant::now();
		let dur = now - self.last_frame_presented;
		self.last_frame_presented = now;
		self.frame_time = dur;

		Ok(())
	}

	/// Submit all the command buffers for this frame to actually render them to the image.
	pub fn submit_frame(
		&mut self, 
		command_buffers: Vec<SecondaryAutoCommandBuffer>,
		transparency_cb: SecondaryAutoCommandBuffer,
	) -> Result<(), GenericEngineError>
	{
		let mut rp_begin_info = RenderPassBeginInfo::framebuffer(self.swapchain.get_current_framebuffer().unwrap());
		rp_begin_info.clear_values = vec![None, None];

		let mut transparency_rp_info = RenderPassBeginInfo::framebuffer(self.swapchain.get_transparency_fb());
		transparency_rp_info.clear_values = vec![
			Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),	// accum
			Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])),	// revealage
			None,	// depth; just load it
		];

		let mut compositing_rp_info = rp_begin_info.clone();
		compositing_rp_info.render_pass = self.swapchain.compositing_rp();

		// finalize the rendering for this frame by executing the secondary command buffers
		let mut primary_cb_builder = AutoCommandBufferBuilder::primary(
			self.graphics_queue.device().clone(),
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
		)?;

		primary_cb_builder
			.begin_render_pass(rp_begin_info.clone(), SubpassContents::SecondaryCommandBuffers)?
			.execute_commands_from_vec(command_buffers)?
			.end_render_pass()?
			.begin_render_pass(transparency_rp_info, SubpassContents::SecondaryCommandBuffers)?
			.execute_commands(transparency_cb)?
			.end_render_pass()?
			.begin_render_pass(compositing_rp_info, SubpassContents::Inline)?
			.set_viewport(0, [self.swapchain.get_viewport()]);
		self.swapchain.bind_for_transparency_compositing(&mut primary_cb_builder)?;
		primary_cb_builder
			.draw(3, 1, 0, 0)?
			.end_render_pass()?;

		self.submit_commands(primary_cb_builder.build()?)?;

		Ok(())
	}

	pub fn get_pipeline(&self, name: &str) -> Result<&pipeline::Pipeline, PipelineNotLoaded>
	{
		Ok(self.material_pipelines.get(name).ok_or(PipelineNotLoaded)?)
	}

	pub fn swapchain_dimensions(&self) -> [u32; 2]
	{
		self.swapchain.dimensions()
	}

	/// Get the current swapchain framebuffer.
	pub fn get_current_framebuffer(&self) -> Option<Arc<Framebuffer>>
	{
		self.swapchain.get_current_framebuffer()
	}
	pub fn get_transparency_framebuffer(&self) -> Arc<Framebuffer>
	{
		self.swapchain.get_transparency_fb()
	}
	pub fn get_swapchain_render_pass(&self) -> Arc<RenderPass>
	{
		self.swapchain.render_pass()
	}
	pub fn get_swapchain_transparency_rp(&self) -> Arc<RenderPass>
	{
		self.swapchain.transparency_rp()
	}

	pub fn get_queue(&self) -> Arc<Queue>
	{
		self.graphics_queue.clone()
	}
	pub fn get_surface(&self) -> Arc<vulkano::swapchain::Surface<winit::window::Window>>
	{
		self.swapchain.get_surface()
	}

	/// Get the delta time for last frame.
	pub fn delta(&self) -> std::time::Duration
	{
		self.frame_time
	}
}

/// Bind the given descriptor sets to the currently bound pipeline on the given command buffer builder.
/// This will fail if there is no pipeline currently bound.
pub fn bind_descriptor_set<L, S>(
	cb: &mut AutoCommandBufferBuilder<L>, first_set: u32, descriptor_sets: S,
) -> Result<(), PipelineExecutionError>
where
	S: DescriptorSetsCollection,
{
	let pipeline_layout = cb
		.state()
		.pipeline_graphics()
		.ok_or(PipelineExecutionError::PipelineNotBound)?
		.layout()
		.clone();
	cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout, first_set, descriptor_sets);
	Ok(())
}

#[derive(Debug)]
pub struct PipelineNotLoaded;
impl std::error::Error for PipelineNotLoaded {}
impl std::fmt::Display for PipelineNotLoaded
{
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
	{
		write!(f, "the specified pipeline is not loaded")
	}
}

enum StagingWork
{
	CopyBuffer(CopyBufferInfo),
	CopyBufferToImage(CopyBufferToImageInfo),
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
		return ((version >> 22) & 0x3ff, (version >> 14) & 0x0ff, (version >> 6) & 0x0ff, version & 0x003f);
	}

	// Intel (Windows only)
	#[cfg(target_family = "windows")]
	if vendor_id == 0x8086 {
		return ((version >> 14), version & 0x3fff, 0, 0);
	}

	// others (use Vulkan version convention)
	((version >> 22), (version >> 12) & 0x3ff, version & 0xfff, 0)
}
fn print_physical_devices(vkinst: &Arc<vulkano::instance::Instance>) -> Result<(), vulkano::VulkanError>
{
	log::info!("Available Vulkan physical devices:");
	for (i, pd) in vkinst.enumerate_physical_devices()?.enumerate() {
		let driver_ver = decode_driver_version(pd.properties().driver_version, pd.properties().vendor_id);

		log::info!(
			"{}: {} ({:?}), driver '{}' version {}.{}.{}.{} (Vulkan {})",
			i,
			pd.properties().device_name,
			pd.properties().device_type,
			pd.properties()
				.driver_name
				.clone()
				.unwrap_or("unknown driver".into()),
			driver_ver.0,
			driver_ver.1,
			driver_ver.2,
			driver_ver.3,
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

/// Get the most appropriate GPU, along with a pair of graphics queue family and transfer queue family.
fn get_physical_device(
	vkinst: Arc<vulkano::instance::Instance>,
) -> Result<(Arc<PhysicalDevice>, Vec<usize>), GenericEngineError>
{
	print_physical_devices(&vkinst)?;
	let dgpu = vkinst
		.enumerate_physical_devices()?
		.find(|pd| pd.properties().device_type == PhysicalDeviceType::DiscreteGpu);
	let igpu = vkinst
		.enumerate_physical_devices()?
		.find(|pd| pd.properties().device_type == PhysicalDeviceType::IntegratedGpu);

	// Try to use a discrete GPU. If there is no discrete GPU, use an integrated GPU instead.
	let physical_device = dgpu.or(igpu).ok_or("No GPUs were found!")?;

	log::info!("Using physical device: {}", physical_device.properties().device_name);

	// get queue family that supports graphics
	print_queue_families(physical_device.queue_family_properties());
	let (graphics_qf, _) = physical_device
		.queue_family_properties()
		.iter()
		.enumerate()
		.find(|(_, q)| q.queue_flags.graphics)
		.ok_or("No graphics queue family found!")?;
	// TODO: Enable the transfer queue again after the next vulkano release.
	// As of vulkano 0.31, vulkano has a bug where the number of queue families
	// used by buffers and images isn't given to `ash`.
	/*let transfer_qf = None;physical_device.queue_family_properties()
		.iter()
		.enumerate()
		.find(|(_, q)| !q.queue_flags.graphics && q.queue_flags.transfer);

	let queue_families = match transfer_qf {
		Some((tqf, _)) => vec![ graphics_qf, tqf ],
		None => vec![ graphics_qf ]
	};*/
	let queue_families = vec![graphics_qf];

	Ok((physical_device, queue_families))
}

/// Set up the Vulkan instance, physical device, logical device, and queue.
/// Returns a graphics queue (which owns the device), and an optional transfer queue.
fn vulkan_setup(game_name: &str) -> Result<(Arc<Queue>, Option<Arc<Queue>>), GenericEngineError>
{
	let vkinst = create_vulkan_instance(game_name)?;
	let (physical_device, queue_families) = get_physical_device(vkinst.clone())?;

	// Select features and extensions.
	// The ones chosen here are practically universally supported by any device with Vulkan support.
	let dev_features = vulkano::device::Features {
		image_cube_array: true,
		independent_blend: true,
		sampler_anisotropy: true,
		texture_compression_bc: true, // change this to ASTC or ETC2 if we want to support mobile platforms
		geometry_shader: true,
		..vulkano::device::Features::empty()
	};
	let dev_extensions = vulkano::device::DeviceExtensions {
		khr_swapchain: true,
		..vulkano::device::DeviceExtensions::empty()
	};

	let queue_create_infos: Vec<_> = queue_families
		.iter()
		.map(|qf| QueueCreateInfo {
			queue_family_index: (*qf).try_into().unwrap(),
			..Default::default()
		})
		.collect();
	let dev_create_info = vulkano::device::DeviceCreateInfo {
		enabled_extensions: dev_extensions,
		enabled_features: dev_features,
		queue_create_infos,
		..Default::default()
	};

	let (_, mut queues) = vulkano::device::Device::new(physical_device, dev_create_info)?;
	let graphics_queue = queues
		.nth(0)
		.ok_or("`vulkano::device::Device::new(...) returned 0 queues`")?;
	let transfer_queue = queues.nth(1);
	Ok((graphics_queue, transfer_queue))
}
