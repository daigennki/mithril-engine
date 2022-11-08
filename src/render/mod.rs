/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod model;
pub mod pipeline;
pub mod skybox;
mod swapchain;
pub mod texture;
pub mod transparency;

use std::collections::HashMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer, TypedBufferAccess};
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, CommandBufferBeginError, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderPassInfo,
	CommandBufferInheritanceRenderPassType, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo, PipelineExecutionError,
	PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryAutoCommandBuffer, SubpassContents, BlitImageInfo,
	PrimaryCommandBufferAbstract, allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
};
use vulkano::descriptor_set::{
	DescriptorSetsCollection, PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator
};
use vulkano::device::{
	physical::{PhysicalDevice, PhysicalDeviceType},
	Queue, QueueCreateInfo, QueueFamilyProperties,
};
use vulkano::format::{ClearValue, Format};
use vulkano::image::{AttachmentImage, ImageDimensions, ImageUsage, MipmapsCount, SwapchainImage, view::ImageView};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{graphics::viewport::Viewport, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::sync::GpuFuture;
use winit::window::WindowBuilder;

use crate::GenericEngineError;
use model::Model;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	swapchain: swapchain::Swapchain,
	graphics_queue: Arc<Queue>,         // this also owns the logical device
	transfer_queue: Option<Arc<Queue>>, // if there is a separate (preferably dedicated) transfer queue, use it for transfers
	descriptor_set_allocator: StandardDescriptorSetAllocator,
	memory_allocator: Arc<StandardMemoryAllocator>,
	command_buffer_allocator: StandardCommandBufferAllocator,

	// Futures from submitted immutable buffer/image transfers. Only used if a separate transfer queue exists.
	transfer_future: Option<Box<dyn GpuFuture + Send + Sync>>,

	// Loaded 3D models, with the key being the path relative to the current working directory.
	models: HashMap<PathBuf, Arc<Model>>,

	// User-accessible material pipelines; these will have their viewports resized
	// when the window size changes
	material_pipelines: HashMap<String, pipeline::Pipeline>,

	// An FP16, linear gamma framebuffer which everything will be rendered to.
	// The final contents of this framebuffer's color image will be blitted to the swapchain's image.
	main_framebuffer: Arc<Framebuffer>,
	color_image: Arc<AttachmentImage>,
	depth_image: Arc<AttachmentImage>,

	transparency_renderer: transparency::TransparencyRenderer,
	
	// TODO: put non-material shaders (shadow filtering, post processing) into different containers
	last_frame_presented: std::time::Instant,
	frame_time: std::time::Duration,

	resize_this_frame: bool,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) -> Result<Self, GenericEngineError>
	{
		let (graphics_queue, transfer_queue) = vulkan_setup(game_name)?;

		// create window
		let window = WindowBuilder::new()
			.with_min_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
			.with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)) // TODO: load this from config
			.with_title(game_name)
			//.with_resizable(false)
			.build(&event_loop)?;

		let vk_dev = graphics_queue.device().clone();
		let swapchain = swapchain::Swapchain::new(vk_dev.clone(), window)?;

		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(vk_dev.clone());
		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_dev.clone()));

		// TODO: we might not need very many primary command buffers here
		let command_buffer_allocator = StandardCommandBufferAllocator::new(
			vk_dev.clone(), StandardCommandBufferAllocatorCreateInfo::default()
		);

		let main_rp = vulkano::single_pass_renderpass!(
			vk_dev.clone(),
			attachments: {
				color: {
					load: DontCare,	// this is DontCare since drawing the skybox effectively clears the image for us
					store: Store,
					format: Format::R16G16B16A16_SFLOAT,
					samples: 1,
				},
				depth: {
					load: DontCare,	// this too is DontCare since the skybox clears it with 1.0
					store: Store,	// order-independent transparency needs this to be `Store`
					format: Format::D16_UNORM,	// NOTE: 24-bit depth formats are unsupported on a significant number of GPUs
					samples: 1,
				}
			},
			pass: {
				color: [color],
				depth_stencil: {depth}
			}
		)?;

		let color_usage = ImageUsage { transfer_src: true, ..Default::default() };
		let color_image = AttachmentImage::with_usage(
			&memory_allocator, swapchain.dimensions(), Format::R16G16B16A16_SFLOAT, color_usage
		)?;
		let depth_image = AttachmentImage::new(&memory_allocator, swapchain.dimensions(), Format::D16_UNORM)?;
		let fb_create_info = FramebufferCreateInfo {
			attachments: vec![
				ImageView::new_default(color_image.clone())?,
				ImageView::new_default(depth_image.clone())?,
			],
			..Default::default()
		};
		let main_framebuffer = Framebuffer::new(main_rp.clone(), fb_create_info)?;

		let transparency_renderer = transparency::TransparencyRenderer::new(
			&memory_allocator, &descriptor_set_allocator, depth_image.clone()
		)?;

		Ok(RenderContext {
			swapchain,
			graphics_queue,
			transfer_queue,
			descriptor_set_allocator,
			memory_allocator,
			command_buffer_allocator,
			transfer_future: None,
			models: HashMap::new(),
			material_pipelines: HashMap::new(),
			main_framebuffer,
			color_image,
			depth_image,
			transparency_renderer,
			last_frame_presented: std::time::Instant::now(),
			frame_time: std::time::Duration::ZERO,
			resize_this_frame: false,
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
		let transparency_rp = self.get_transparency_framebuffer().render_pass().clone();
		self.material_pipelines
			.insert(name, pipeline::Pipeline::new_from_yaml(
				filename, 
				self.get_main_render_pass().first_subpass(), 
				Some(transparency_rp.first_subpass())
			)?);
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
		let (tex, staging_work) = texture::Texture::new(&self.memory_allocator, path)?;
		self.submit_transfer(staging_work.into())?;
		Ok(tex)
	}

	pub fn new_cubemap_texture(&mut self, faces: [PathBuf; 6]) -> Result<texture::CubemapTexture, GenericEngineError>
	{
		let (tex, staging_work) = texture::CubemapTexture::new(&self.memory_allocator, faces)?;
		self.submit_transfer(staging_work.into())?;
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
		let (tex, staging_work) = texture::Texture::new_from_iter(&self.memory_allocator, iter, vk_fmt, dimensions, mip)?;
		self.submit_transfer(staging_work.into())?;
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
		let staging_buf = CpuAccessibleBuffer::from_iter(
			&self.memory_allocator, staging_usage, false, data
		)?;
		usage.transfer_dst = true;
		let buf = DeviceLocalBuffer::array(
			&self.memory_allocator,
			staging_buf.len(),
			usage,
			self.get_queue_families(),
		)?;
		self.submit_transfer(CopyBufferInfo::buffers(staging_buf, buf.clone()).into())?;
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
		let staging_buf = CpuAccessibleBuffer::from_data(&self.memory_allocator, staging_usage, false, data)?;
		usage.transfer_dst = true;
		let buf = DeviceLocalBuffer::new(&self.memory_allocator, usage, self.get_queue_families())?;
		self.submit_transfer(CopyBufferInfo::buffers(staging_buf, buf.clone()).into())?;
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
	) -> Result<(CpuBufferPool<T>, Arc<DeviceLocalBuffer<T>>), GenericEngineError>
	where
		[T]: vulkano::buffer::BufferContents,
		T: Send + Sync + bytemuck::Pod,
	{
		let cpu_buf = CpuBufferPool::upload(self.memory_allocator.clone());
		usage.transfer_dst = true;
		let gpu_buf = DeviceLocalBuffer::new(&self.memory_allocator, usage, self.get_queue_families())?;
		self.submit_transfer_on_graphics_queue(CopyBufferInfo::buffers(cpu_buf.from_data(data)?, gpu_buf.clone()).into())?;
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
		-> Result<(), GenericEngineError>
	{
		self.submit_transfer_on_graphics_queue(CopyBufferInfo::buffers(src, dst).into())
	}

	pub fn descriptor_set_allocator(&self) -> &StandardDescriptorSetAllocator
	{
		&self.descriptor_set_allocator
	}
	pub fn memory_allocator(&self) -> &StandardMemoryAllocator
	{
		&self.memory_allocator
	}

	pub fn new_descriptor_set(
		&self, pipeline_name: &str, set: usize, writes: impl IntoIterator<Item = WriteDescriptorSet>,
	) -> Result<Arc<PersistentDescriptorSet>, GenericEngineError>
	{
		self.material_pipelines
			.get(pipeline_name)
			.ok_or(PipelineNotLoaded)?
			.new_descriptor_set(&self.descriptor_set_allocator, set, writes)
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
			&self.command_buffer_allocator,
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

	/// Update images to match the current window size.
	fn fit_images_to_window(&mut self) -> Result<(), GenericEngineError>
	{
		let color_usage = ImageUsage { transfer_src: true,..Default::default() };
		self.color_image = AttachmentImage::with_usage(
			&self.memory_allocator, self.swapchain.dimensions(), Format::R16G16B16A16_SFLOAT, color_usage
		)?;

		let depth_usage = ImageUsage { sampled: true, ..Default::default() };
		self.depth_image = AttachmentImage::with_usage(
			&self.memory_allocator, self.swapchain.dimensions(), Format::D16_UNORM, depth_usage
		)?;

		let fb_create_info = FramebufferCreateInfo {
			attachments: vec![
				ImageView::new_default(self.color_image.clone())?,
				ImageView::new_default(self.depth_image.clone())?,
			],
			..Default::default()
		};
		self.main_framebuffer = Framebuffer::new(self.main_framebuffer.render_pass().clone(), fb_create_info)?;

		self.transparency_renderer.resize_image(
			&self.memory_allocator, &self.descriptor_set_allocator, self.depth_image.clone()
		)?;

		Ok(())
	}

	/// Tell the swapchain to go to the next image.
	/// The image size *may* change here.
	/// This must only be called once per frame.
	///
	/// This returns the acquired swapchain image.
	fn next_swapchain_image(&mut self) -> Result<Arc<SwapchainImage>, GenericEngineError>
	{
		let (image, dimensions_changed) = self.swapchain.get_next_image()?;
		if dimensions_changed {
			self.fit_images_to_window()?;
		}
		self.resize_this_frame = dimensions_changed;
		
		Ok(image)
	}

	/// Submit staging work for immutable objects to the transfer queue, or if there is no transfer queue, 
	/// keep it for later when the graphics operations are submitted.
	fn submit_transfer(&mut self, work: StagingWork) -> Result<(), GenericEngineError>
	{
		match self.transfer_queue.as_ref() {
			Some(q) => {
				let mut staging_cb_builder = AutoCommandBufferBuilder::primary(
					&self.command_buffer_allocator,
					q.queue_family_index(),
					CommandBufferUsage::OneTimeSubmit,
				)?;
				match work {
					StagingWork::CopyBuffer(info) => staging_cb_builder.copy_buffer(info)?,
					StagingWork::CopyBufferToImage(info) => staging_cb_builder.copy_buffer_to_image(info)?,
				};
				let staging_cb = staging_cb_builder.build()?;

				let new_future = match self.transfer_future.take() {
					Some(f) => staging_cb.execute_after(f, q.clone())?.boxed_send_sync(),
					None => staging_cb.execute(q.clone())?.boxed_send_sync(),
				};
				new_future.flush()?;
				self.transfer_future = Some(new_future);
			},
			None => self.submit_transfer_on_graphics_queue(work)?,
		}
		Ok(())
	}

	/// Submit staging work for *mutable* objects to the graphics queue. Use this instead of `submit_transfer` if 
	/// there's the possibility that the object is in use by a previous submission.
	fn submit_transfer_on_graphics_queue(&mut self, work: StagingWork) -> Result<(), GenericEngineError>
	{
		let mut staging_cb_builder = AutoCommandBufferBuilder::primary(
			&self.command_buffer_allocator,
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
		)?;
		match work {
			StagingWork::CopyBuffer(info) => staging_cb_builder.copy_buffer(info)?,
			StagingWork::CopyBufferToImage(info) => staging_cb_builder.copy_buffer_to_image(info)?,
		};
		let staging_cb = staging_cb_builder.build()?;

		self.swapchain.submit_transfer_on_graphics_queue(staging_cb, self.graphics_queue.clone())?;
		Ok(())
	}

	fn submit_commands(&mut self, built_cb: PrimaryAutoCommandBuffer) -> Result<(), GenericEngineError>
	{
		self.swapchain.submit_commands(
			built_cb,
			self.graphics_queue.clone(),
			self.transfer_future.take(),
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
		let mut rp_begin_info = RenderPassBeginInfo::framebuffer(self.main_framebuffer.clone());
		rp_begin_info.clear_values = vec![None, None];

		let mut transparency_rp_info = RenderPassBeginInfo::framebuffer(self.transparency_renderer.framebuffer());
		transparency_rp_info.clear_values = vec![
			Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),	// accum
			Some(ClearValue::Float([1.0, 0.0, 0.0, 0.0])),	// revealage
			None,	// depth; just load it
		];

		// finalize the rendering for this frame by executing the secondary command buffers
		let mut primary_cb_builder = AutoCommandBufferBuilder::primary(
			&self.command_buffer_allocator,
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
		)?;

		primary_cb_builder
			.begin_render_pass(rp_begin_info.clone(), SubpassContents::SecondaryCommandBuffers)?
			.execute_commands_from_vec(command_buffers)?
			.end_render_pass()?
			.begin_render_pass(transparency_rp_info, SubpassContents::SecondaryCommandBuffers)?
			.execute_commands(transparency_cb)?
			.end_render_pass()?;
			
		self.transparency_renderer.composite_transparency(&mut primary_cb_builder, self.main_framebuffer.clone())?;

		let blit_info = BlitImageInfo::images(self.color_image.clone(), self.next_swapchain_image()?);
		primary_cb_builder.blit_image(blit_info)?;
		self.submit_commands(primary_cb_builder.build()?)?;

		Ok(())
	}

	/// Check if the window has been resized since the last frame submission.
	pub fn window_resized(&self) -> bool
	{
		self.resize_this_frame
	}

	pub fn swapchain_dimensions(&self) -> [u32; 2]
	{
		self.swapchain.dimensions()
	}

	pub fn get_pipeline(&self, name: &str) -> Result<&pipeline::Pipeline, PipelineNotLoaded>
	{
		Ok(self.material_pipelines.get(name).ok_or(PipelineNotLoaded)?)
	}

	pub fn get_main_framebuffer(&self) -> Arc<Framebuffer>
	{
		self.main_framebuffer.clone()
	}
	pub fn get_transparency_framebuffer(&self) -> Arc<Framebuffer>
	{
		self.transparency_renderer.framebuffer()
	}
	pub fn get_main_render_pass(&self) -> Arc<RenderPass>
	{
		self.main_framebuffer.render_pass().clone()
	}

	pub fn get_queue(&self) -> Arc<Queue>
	{
		self.graphics_queue.clone()
	}
	pub fn get_surface(&self) -> Arc<vulkano::swapchain::Surface>
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

	// Get a queue family that supports graphics operations.
	print_queue_families(physical_device.queue_family_properties());
	let (graphics_qf, _) = physical_device
		.queue_family_properties()
		.iter()
		.enumerate()
		.find(|(_, q)| q.queue_flags.graphics)
		.ok_or("No graphics queue family found!")?;

	// Get a separate queue family for transfers.
	// First try to get one that is specifically optimized for transfers (supports netiher graphics nor compute),
	// then if such a queue family doesn't exist, use one that just doesn't support graphics.
	let transfer_qf = physical_device.queue_family_properties()
		.iter()
		.enumerate()
		.find(|(_, q)| !q.queue_flags.graphics && !q.queue_flags.compute && q.queue_flags.transfer)
		.or_else(|| {
			physical_device.queue_family_properties()
				.iter()
				.enumerate()
				.find(|(_, q)| !q.queue_flags.graphics && q.queue_flags.transfer)
		});

	let queue_families = match transfer_qf {
		Some((tqf, _)) => {
			log::info!("Using queue family {} for transfers", tqf);
			vec![ graphics_qf, tqf ]
		},
		None => vec![ graphics_qf ]
	};

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
		.next()
		.ok_or("`vulkano::device::Device::new(...) returned 0 queues`")?;
	let transfer_queue = queues.next();
	Ok((graphics_queue, transfer_queue))
}
