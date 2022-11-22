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
mod vulkan_init;

use std::collections::HashMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use vulkano::buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer, TypedBufferAccess};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	AutoCommandBufferBuilder, BlitImageInfo, CommandBufferBeginError, CommandBufferInheritanceInfo,
	CommandBufferInheritanceRenderPassInfo, CommandBufferInheritanceRenderPassType, CommandBufferUsage, CopyBufferInfo,
	CopyBufferToImageInfo, PipelineExecutionError, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
	SecondaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::{
	allocator::StandardDescriptorSetAllocator, DescriptorSetsCollection, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{DeviceOwned, Queue};
use vulkano::format::Format;
use vulkano::image::{view::ImageView, AttachmentImage, ImageDimensions, ImageUsage, MipmapsCount, SwapchainImage};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{graphics::viewport::Viewport, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::sampler::{Sampler, SamplerCreateInfo};
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

	// Samplers used for 3D draws. All use linear downscaling and have 16x anisotropic filtering enabled.
	sampler_linear: Arc<Sampler>, // Linear upscaling (default)
	//sampler_nearest: Arc<Sampler>, // Nearest neighbor upscaling (possibly useful for pixel art)

	trm: Mutex<ThreadedRenderingManager>,

	// Futures from submitted immutable buffer/image transfers. Only used if a separate transfer queue exists.
	transfer_future: Option<Box<dyn GpuFuture + Send + Sync>>,

	// Loaded 3D models, with the key being the path relative to the current working directory.
	models: HashMap<PathBuf, Arc<Model>>,

	// User-accessible material pipelines; these will have their viewports resized
	// when the window size changes
	material_pipelines: HashMap<String, pipeline::Pipeline>,

	// The final contents of this render target's color image will be blitted to the swapchain's image.
	main_render_target: RenderTarget,

	transparency_renderer: transparency::MomentTransparencyRenderer,

	// TODO: put non-material shaders (shadow filtering, post processing) into different containers
	last_frame_presented: std::time::Instant,
	frame_time: std::time::Duration,

	resize_this_frame: bool,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) -> Result<Self, GenericEngineError>
	{
		let (graphics_queue, transfer_queue) = vulkan_init::vulkan_setup(game_name)?;

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
		let command_buffer_allocator =
			StandardCommandBufferAllocator::new(vk_dev.clone(), StandardCommandBufferAllocatorCreateInfo::default());

		let sampler_info = SamplerCreateInfo {
			anisotropy: Some(16.0),
			..SamplerCreateInfo::simple_repeat_linear()
		};
		let sampler_linear = Sampler::new(vk_dev.clone(), sampler_info)?;

		let main_render_target = RenderTarget::new(&memory_allocator, swapchain.dimensions())?;
		let transparency_renderer = transparency::MomentTransparencyRenderer::new(
			&memory_allocator,
			&descriptor_set_allocator,
			main_render_target.depth_image().clone(),
			sampler_linear.clone()
		)?;

		Ok(RenderContext {
			swapchain,
			graphics_queue,
			transfer_queue,
			descriptor_set_allocator,
			memory_allocator,
			command_buffer_allocator,
			sampler_linear,
			trm: Mutex::new(ThreadedRenderingManager::new(8)),
			transfer_future: None,
			models: HashMap::new(),
			material_pipelines: HashMap::new(),
			main_render_target,
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
		let transparency_rp = self.transparency_renderer.framebuffer().render_pass().clone();
		self.material_pipelines.insert(
			name,
			pipeline::Pipeline::new_from_yaml(
				filename,
				self.get_main_render_pass().first_subpass(),
				Some(transparency_rp.first_subpass()),
				self.sampler_linear.clone()
			)?,
		);
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
		let staging_buf = CpuAccessibleBuffer::from_iter(&self.memory_allocator, staging_usage, false, data)?;
		usage.transfer_dst = true;
		let buf = DeviceLocalBuffer::array(&self.memory_allocator, staging_buf.len(), usage, self.get_queue_families())?;
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
		self.copy_buffer(cpu_buf.from_data(data)?, gpu_buf.clone())?;
		Ok((cpu_buf, gpu_buf))
	}

	fn get_queue_families(&self) -> Vec<u32>
	{
		self.graphics_queue
			.device()
			.active_queue_family_indices()
			.into()
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
	fn new_secondary_command_buffer(
		&self, framebuffer: Arc<Framebuffer>, render_pass: Option<Arc<RenderPass>>
	) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, CommandBufferBeginError>
	{
		let use_rp = render_pass
			.unwrap_or_else(|| framebuffer.render_pass().clone());
		let inherit_rp = CommandBufferInheritanceRenderPassType::BeginRenderPass(CommandBufferInheritanceRenderPassInfo {
			subpass: use_rp.first_subpass(),
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
			origin: [0.0, 0.0],
			dimensions: [
				fb_extent[0] as f32, fb_extent[1] as f32,
			],
			depth_range: 0.0..1.0,
		};
		cb.set_viewport(0, [viewport]);

		Ok(cb)
	}
	pub fn record_main_draws(&self) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, CommandBufferBeginError>
	{
		self.new_secondary_command_buffer(self.main_render_target.framebuffer().clone(), None)
	}
	pub fn record_transparency_moments_draws(
		&self
	)-> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, CommandBufferBeginError>
	{
		self.new_secondary_command_buffer(self.transparency_renderer.moments_framebuffer(), None)
	}
	pub fn record_transparency_draws(
		&self
	) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, CommandBufferBeginError>
	{
		self.new_secondary_command_buffer(self.transparency_renderer.framebuffer(), None)
	}
	pub fn record_ui_draws(
		&self
	) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, CommandBufferBeginError>
	{
		self.new_secondary_command_buffer(
			self.main_render_target.framebuffer().clone(), 
			Some(self.main_render_target.ui_rp().clone())
		)
	}

	pub fn get_moments_pl(&self) -> &pipeline::Pipeline
	{
		self.transparency_renderer.get_moments_pipeline()
	}
	pub fn get_moments_set(&self) -> Arc<PersistentDescriptorSet>
	{
		self.transparency_renderer.get_moments_descriptor_set()
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
			// Update images to match the current window size.
			self.main_render_target
				.resize(&self.memory_allocator, self.swapchain.dimensions())?;

			self.transparency_renderer.resize_image(
				&self.memory_allocator,
				&self.descriptor_set_allocator,
				self.main_render_target.depth_image().clone(),
			)?;
		}
		self.resize_this_frame = dimensions_changed;

		Ok(image)
	}

	/// Create a command buffer for a transfer.
	/// TODO: should we wait for more work before we build a command buffer? it's probably inefficient to submit a command
	/// buffer with just a single command...
	fn create_staging_command_buffer(
		&self, work: StagingWork, queue_family: u32,
	) -> Result<PrimaryAutoCommandBuffer, GenericEngineError>
	{
		let mut staging_cb_builder =
			AutoCommandBufferBuilder::primary(&self.command_buffer_allocator, queue_family, CommandBufferUsage::OneTimeSubmit)?;
		match work {
			StagingWork::CopyBuffer(info) => staging_cb_builder.copy_buffer(info)?,
			StagingWork::CopyBufferToImage(info) => staging_cb_builder.copy_buffer_to_image(info)?,
		};
		Ok(staging_cb_builder.build()?)
	}

	/// Submit staging work for immutable objects to the transfer queue, or if there is no transfer queue,
	/// keep it for later when the graphics operations are submitted.
	fn submit_transfer(&mut self, work: StagingWork) -> Result<(), GenericEngineError>
	{
		match self.transfer_queue.as_ref() {
			Some(q) => {
				let staging_cb = self.create_staging_command_buffer(work, q.queue_family_index())?;
				let new_future = match self.transfer_future.take() {
					Some(f) => staging_cb.execute_after(f, q.clone())?.boxed_send_sync(),
					None => staging_cb.execute(q.clone())?.boxed_send_sync(),
				};
				new_future.flush()?;
				self.transfer_future = Some(new_future);
			}
			None => self.submit_transfer_on_graphics_queue(work)?,
		}
		Ok(())
	}

	/// Submit staging work for *mutable* objects to the graphics queue. Use this instead of `submit_transfer` if
	/// there's the possibility that the object is in use by a previous submission.
	fn submit_transfer_on_graphics_queue(&mut self, work: StagingWork) -> Result<(), GenericEngineError>
	{
		let staging_cb = self.create_staging_command_buffer(work, self.graphics_queue.queue_family_index())?;
		self.swapchain
			.submit_transfer_on_graphics_queue(staging_cb, self.graphics_queue.clone())?;
		Ok(())
	}

	/// Queue a buffer copy which will be executed before the next image submission.
	/// Basically a shortcut for `submit_transfer_on_graphics_queue`.
	pub fn copy_buffer(&mut self, src: Arc<dyn BufferAccess>, dst: Arc<dyn BufferAccess>) -> Result<(), GenericEngineError>
	{
		self.submit_transfer_on_graphics_queue(CopyBufferInfo::buffers(src, dst).into())
	}

	fn lock_trm(&self) -> Result<std::sync::MutexGuard<'_, ThreadedRenderingManager>, ThreadedRenderingLockError>
	{
		let lock_guard = self
			.trm
			.lock()
			.map_err(|e| ThreadedRenderingLockError::new(e))?;
		Ok(lock_guard)
	}
	pub fn add_cb(&self, cb: SecondaryAutoCommandBuffer) -> Result<(), ThreadedRenderingLockError>
	{
		self.lock_trm()?.add_cb(cb);
		Ok(())
	}
	pub fn add_transparency_moments_cb(&self, cb: SecondaryAutoCommandBuffer) -> Result<(), ThreadedRenderingLockError>
	{
		self.lock_trm()?.add_transparency_moments_cb(cb);
		Ok(())
	}
	pub fn add_transparency_cb(&self, cb: SecondaryAutoCommandBuffer) -> Result<(), ThreadedRenderingLockError>
	{
		self.lock_trm()?.add_transparency_cb(cb);
		Ok(())
	}
	pub fn add_ui_cb(&self, cb: SecondaryAutoCommandBuffer) -> Result<(), ThreadedRenderingLockError>
	{
		self.lock_trm()?.add_ui_cb(cb);
		Ok(())
	}

	fn submit_commands(&mut self, built_cb: PrimaryAutoCommandBuffer) -> Result<(), GenericEngineError>
	{
		self.swapchain
			.present(built_cb, self.graphics_queue.clone(), self.transfer_future.take())?;

		let now = std::time::Instant::now();
		let dur = now - self.last_frame_presented;
		self.last_frame_presented = now;
		self.frame_time = dur;

		Ok(())
	}

	/// Submit all the command buffers for this frame to actually render them to the image.
	pub fn submit_frame(&mut self) -> Result<(), GenericEngineError>
	{
		let command_buffers;
		let transparency_moments_cb;
		let transparency_cb;
		let ui_cb;
		{
			let mut trm_locked = self.lock_trm()?;
			command_buffers = trm_locked.take_built_command_buffers();
			transparency_moments_cb = trm_locked.take_transparency_moments_cb().unwrap();
			transparency_cb = trm_locked.take_transparency_cb().unwrap();
			ui_cb = trm_locked.take_ui_cb();
		}

		let mut rp_begin_info = RenderPassBeginInfo::framebuffer(self.main_render_target.framebuffer().clone());
		rp_begin_info.clear_values = vec![None, None];

		// finalize the rendering for this frame by executing the secondary command buffers
		let mut primary_cb_builder = AutoCommandBufferBuilder::primary(
			&self.command_buffer_allocator,
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
		)?;

		primary_cb_builder
			.begin_render_pass(rp_begin_info.clone(), SubpassContents::SecondaryCommandBuffers)?
			.execute_commands_from_vec(command_buffers)?
			.end_render_pass()?;

		self.transparency_renderer.process_transparency(
			transparency_moments_cb,
			transparency_cb,
			&mut primary_cb_builder,
			self.main_render_target.framebuffer().clone(),
		)?;

		let blit_info = BlitImageInfo::images(self.main_render_target.color_image().clone(), self.next_swapchain_image()?);
		rp_begin_info.render_pass = self.main_render_target.ui_rp().clone();
		primary_cb_builder
			.begin_render_pass(rp_begin_info, SubpassContents::SecondaryCommandBuffers)?
			.execute_commands_from_vec(ui_cb)?
			.end_render_pass()?
			.blit_image(blit_info)?;
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

	pub fn get_main_render_pass(&self) -> Arc<RenderPass>
	{
		self.main_render_target.framebuffer().render_pass().clone()
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

#[derive(Debug)]
pub struct ThreadedRenderingLockError
{
	cause: String,
}
impl ThreadedRenderingLockError
{
	pub fn new<T>(e: std::sync::PoisonError<T>) -> Self
	{
		Self { cause: format!("{}", e) }
	}
}
impl std::error::Error for ThreadedRenderingLockError {}
impl std::fmt::Display for ThreadedRenderingLockError
{
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
	{
		write!(f, "failed to lock ThreadedRenderingManager: {}", &self.cause)
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

struct ThreadedRenderingManager
{
	built_command_buffers: Vec<SecondaryAutoCommandBuffer>,
	transparency_moments_cb: Option<SecondaryAutoCommandBuffer>,
	transparency_cb: Option<SecondaryAutoCommandBuffer>,
	ui_cb: Vec<SecondaryAutoCommandBuffer>,
	default_capacity: usize,
}
impl ThreadedRenderingManager
{
	pub fn new(default_capacity: usize) -> Self
	{
		ThreadedRenderingManager {
			built_command_buffers: Vec::with_capacity(default_capacity),
			transparency_moments_cb: None,
			transparency_cb: None,
			ui_cb: Vec::with_capacity(2),
			default_capacity,
		}
	}

	/// Add a secondary command buffer that has been built.
	pub fn add_cb(&mut self, command_buffer: SecondaryAutoCommandBuffer)
	{
		self.built_command_buffers.push(command_buffer);
	}

	pub fn add_transparency_moments_cb(&mut self, command_buffer: SecondaryAutoCommandBuffer)
	{
		self.transparency_moments_cb = Some(command_buffer)
	}
	pub fn add_transparency_cb(&mut self, command_buffer: SecondaryAutoCommandBuffer)
	{
		self.transparency_cb = Some(command_buffer)
	}
	pub fn add_ui_cb(&mut self, command_buffer: SecondaryAutoCommandBuffer)
	{
		self.ui_cb.push(command_buffer)
	}

	/// Take all of the secondary command buffers that have been built.
	pub fn take_built_command_buffers(&mut self) -> Vec<SecondaryAutoCommandBuffer>
	{
		std::mem::replace(&mut self.built_command_buffers, Vec::with_capacity(self.default_capacity))
	}
	pub fn take_transparency_moments_cb(&mut self) -> Option<SecondaryAutoCommandBuffer>
	{
		self.transparency_moments_cb.take()
	}
	pub fn take_transparency_cb(&mut self) -> Option<SecondaryAutoCommandBuffer>
	{
		self.transparency_cb.take()
	}
	pub fn take_ui_cb(&mut self) -> Vec<SecondaryAutoCommandBuffer>
	{
		std::mem::replace(&mut self.ui_cb, Vec::with_capacity(2))
	}
}

struct RenderTarget
{
	// An FP16, linear gamma framebuffer which everything will be rendered to.
	framebuffer: Arc<Framebuffer>,
	color_image: Arc<AttachmentImage>,
	depth_image: Arc<AttachmentImage>,
	ui_rp: Arc<RenderPass>,
}
impl RenderTarget
{
	pub fn new(memory_allocator: &StandardMemoryAllocator, dimensions: [u32; 2]) -> Result<Self, GenericEngineError>
	{
		let vk_dev = memory_allocator.device().clone();
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

		let ui_rp = vulkano::single_pass_renderpass!(
			vk_dev.clone(),
			attachments: {
				color: {
					load: Load,
					store: Store,
					format: Format::R16G16B16A16_SFLOAT,
					samples: 1,
				},
				depth: {
					load: Load,
					store: DontCare,
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
		let color_image = AttachmentImage::with_usage(memory_allocator, dimensions, Format::R16G16B16A16_SFLOAT, color_usage)?;
		let depth_image = AttachmentImage::new(memory_allocator, dimensions, Format::D16_UNORM)?;
		let fb_create_info = FramebufferCreateInfo {
			attachments: vec![
				ImageView::new_default(color_image.clone())?,
				ImageView::new_default(depth_image.clone())?,
			],
			..Default::default()
		};
		let framebuffer = Framebuffer::new(main_rp.clone(), fb_create_info)?;

		Ok(Self { 
			framebuffer, 
			color_image, 
			depth_image, 
			ui_rp 
		})
	}

	pub fn resize(&mut self, memory_allocator: &StandardMemoryAllocator, dimensions: [u32; 2])
		-> Result<(), GenericEngineError>
	{
		let color_usage = ImageUsage { transfer_src: true, ..Default::default() };
		self.color_image = AttachmentImage::with_usage(memory_allocator, dimensions, Format::R16G16B16A16_SFLOAT, color_usage)?;

		let depth_usage = ImageUsage { sampled: true, ..Default::default() };
		self.depth_image = AttachmentImage::with_usage(memory_allocator, dimensions, Format::D16_UNORM, depth_usage)?;

		let fb_create_info = FramebufferCreateInfo {
			attachments: vec![
				ImageView::new_default(self.color_image.clone())?,
				ImageView::new_default(self.depth_image.clone())?,
			],
			..Default::default()
		};
		self.framebuffer = Framebuffer::new(self.framebuffer.render_pass().clone(), fb_create_info)?;

		Ok(())
	}

	pub fn framebuffer(&self) -> &Arc<Framebuffer>
	{
		&self.framebuffer
	}
	pub fn color_image(&self) -> &Arc<AttachmentImage>
	{
		&self.color_image
	}
	pub fn depth_image(&self) -> &Arc<AttachmentImage>
	{
		&self.depth_image
	}
	pub fn ui_rp(&self) -> &Arc<RenderPass>
	{
		&self.ui_rp
	}
}
