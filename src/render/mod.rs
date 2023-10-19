/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
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

use glam::*;

use vulkano::{Validated, VulkanError};
use vulkano::buffer::{
	allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
	subbuffer::Subbuffer,
	Buffer, BufferCreateInfo, BufferUsage,
};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	AutoCommandBufferBuilder, BlitImageInfo, CommandBufferInheritanceInfo,
	CommandBufferInheritanceRenderingInfo, CommandBufferInheritanceRenderPassType, CommandBufferUsage, CopyBufferInfo,
	CopyBufferToImageInfo, CopyImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
	SecondaryAutoCommandBuffer, SubpassContents, RenderingInfo, RenderingAttachmentInfo,
	SecondaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{
	allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{DeviceOwned, Queue};
use vulkano::format::Format;
use vulkano::image::{sampler::{Sampler, SamplerCreateInfo}, view::ImageView, Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{
	graphics::{ subpass::PipelineRenderingCreateInfo, viewport::Viewport }, PipelineBindPoint,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::sync::{GpuFuture, Sharing};
use winit::window::WindowBuilder;

use crate::GenericEngineError;
use texture::Texture;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	swapchain: swapchain::Swapchain,
	graphics_queue: Arc<Queue>,         // this also owns the logical device
	transfer_queue: Option<Arc<Queue>>, // if there is a separate (preferably dedicated) transfer queue, use it for transfers
	descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
	memory_allocator: Arc<StandardMemoryAllocator>,
	command_buffer_allocator: StandardCommandBufferAllocator,

	// Samplers used for 3D draws. All use linear downscaling and have 16x anisotropic filtering enabled.
	sampler_linear: Arc<Sampler>, // Linear upscaling (default)
	//sampler_nearest: Arc<Sampler>, // Nearest neighbor upscaling (possibly useful for pixel art)
	trm: Mutex<ThreadedRenderingManager>,

	// Futures from submitted immutable buffer/image transfers. Only used if a separate transfer queue exists.
	transfer_future: Option<Box<dyn GpuFuture + Send + Sync>>,

	// Loaded textures, with the key being the path relative to the current working directory
	textures: HashMap<PathBuf, Arc<Texture>>,

	// User-accessible material pipelines
	material_pipelines: HashMap<String, pipeline::Pipeline>,

	// The final contents of this render target's color image will be blitted to the intermediate sRGB nonlinear image.
	main_render_target: RenderTarget,

	// An sRGB image which the above `main_render_target` will be blitted to, thus converting it to nonlinear.
	// This will be copied, not blitted, to the swapchain.
	intermediate_srgb_img: Arc<Image>,

	transparency_renderer: transparency::MomentTransparencyRenderer,

	// A common `SubbufferAllocator` for high-frequency staging buffers.
	staging_buffer_allocator: Mutex<SubbufferAllocator>,

	// TODO: put non-material shaders (shadow filtering, post processing) into different containers
	last_frame_presented: std::time::Instant,
	frame_time: std::time::Duration,

	resize_this_frame: bool,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) -> Result<Self, GenericEngineError>
	{
		let (graphics_queue, transfer_queue) = vulkan_init::vulkan_setup(game_name, event_loop)?;

		let use_monitor = event_loop 
			.primary_monitor()
			.or(event_loop.available_monitors().next())
			.ok_or("The primary monitor could not be detected.")?;
		
		let current_video_mode = get_video_modes(use_monitor.clone());

		// attempt to use fullscreen window if requested
		// TODO: load this from config
		let fullscreen = if std::env::args().find(|arg| arg == "-fullscreen").is_some() {
			if std::env::args().find(|arg| arg == "-borderless").is_some() {
				// if "-fullscreen" and "-borderless" were both specified, make a borderless window filling the entire monitor
				// instead of exclusive fullscreen
				Some(winit::window::Fullscreen::Borderless(Some(use_monitor.clone())))
			} else {
				// NOTE: this is specifically *exclusive* fullscreen, which gets ignored on Wayland.
				// therefore, it might be a good idea to hide such an option in UI from the end user on Wayland.
				// TODO: use VK_EXT_full_screen_exclusive to minimize latency (usually only available on Windows)
				if let Some(fullscreen_video_mode) = current_video_mode {
					Some(winit::window::Fullscreen::Exclusive(fullscreen_video_mode))
				} else {
					log::warn!("The current monitor's video mode could not be determined. Fullscreen mode is unavailable.");
					None
				}
			}
		} else {
			None
		};

		let window_size = if let Some(fs_mode) = &fullscreen {
			match fs_mode {
				winit::window::Fullscreen::Exclusive(video_mode) => video_mode.size(),
				winit::window::Fullscreen::Borderless(_) => use_monitor.size()
			}
		} else {
			// TODO: load this from config
			winit::dpi::PhysicalSize::new(1280, 720)
		};

		// create window
		let window = WindowBuilder::new()
			.with_inner_size(window_size)
			.with_title(game_name)
			.with_decorations(std::env::args().find(|arg| arg == "-borderless").is_none())
			.with_fullscreen(fullscreen)
			.build(&event_loop)?;	
		
		let vk_dev = graphics_queue.device().clone();
		let swapchain = swapchain::Swapchain::new(vk_dev.clone(), window)?;

		let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(vk_dev.clone()));
		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_dev.clone()));

		// TODO: we might not need very many primary command buffers here
		let command_buffer_allocator =
			StandardCommandBufferAllocator::new(vk_dev.clone(), StandardCommandBufferAllocatorCreateInfo::default());

		let sampler_info = SamplerCreateInfo {
			anisotropy: Some(16.0),
			..SamplerCreateInfo::simple_repeat_linear()
		};
		let sampler_linear = Sampler::new(vk_dev.clone(), sampler_info)?;

		let main_render_target = RenderTarget::new(memory_allocator.clone(), swapchain.dimensions())?;
		let swapchain_dim = swapchain.dimensions();
		let intermediate_img_create_info = ImageCreateInfo {
			format: Format::B8G8R8A8_SRGB,
			extent: [ swapchain_dim[0], swapchain_dim[1], 1 ],
			usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
			..Default::default()
		};
		let intermediate_srgb_img = 
			Image::new(memory_allocator.clone(), intermediate_img_create_info, AllocationCreateInfo::default())?;
		let transparency_renderer = transparency::MomentTransparencyRenderer::new(
			memory_allocator.clone(),
			descriptor_set_allocator.clone(),
			swapchain.dimensions(),
			sampler_linear.clone(),
		)?;

		let pool_create_info = SubbufferAllocatorCreateInfo {
			arena_size: 128 * 1024, // TODO: determine an appropriate arena size based on actual memory usage
			buffer_usage: BufferUsage::TRANSFER_SRC,
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buffer_allocator = Mutex::new(SubbufferAllocator::new(memory_allocator.clone(), pool_create_info));

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
			textures: HashMap::new(),
			material_pipelines: HashMap::new(),
			main_render_target,
			intermediate_srgb_img,
			transparency_renderer,
			staging_buffer_allocator,
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
		
		let rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![ Some(Format::R16G16B16A16_SFLOAT), ],
			depth_attachment_format: Some(Format::D16_UNORM),
			..Default::default()
		};

		let transparency_weights_rendering = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![
				Some(Format::R16G16B16A16_SFLOAT),
				Some(Format::R8_UNORM),
			],
			depth_attachment_format: Some(Format::D16_UNORM),
			..Default::default()
		};
		self.material_pipelines.insert(
			name,
			pipeline::Pipeline::new_from_yaml(
				filename,
				rendering_info,
				Some(transparency_weights_rendering),
				self.sampler_linear.clone(),
				self.descriptor_set_allocator.clone(),
			)?,
		);
		Ok(())
	}

	pub fn get_default_sampler(&self) -> &Arc<Sampler>
	{
		&self.sampler_linear
	}

	/// Load an image file as a texture into memory.
	/// If the image was already loaded, it'll use the corresponding texture.
	pub fn get_texture(&mut self, path: &Path) -> Result<Arc<Texture>, GenericEngineError>
	{
		match self.textures.get(path) {
			Some(tex) => {
				log::info!("Reusing loaded texture '{}'", path.display());
				Ok(tex.clone())
			}
			None => {
				let (tex, staging_work) = texture::Texture::new(self.memory_allocator.clone(), path)?;
				self.submit_transfer(staging_work.into())?;
				let tex_arc = Arc::new(tex);
				self.textures.insert(path.to_path_buf(), tex_arc.clone());
				Ok(tex_arc)
			}
		}
	}

	pub fn new_cubemap_texture(&mut self, faces: [PathBuf; 6]) -> Result<texture::CubemapTexture, GenericEngineError>
	{
		let (tex, staging_work) = texture::CubemapTexture::new(self.memory_allocator.clone(), faces)?;
		self.submit_transfer(staging_work.into())?;
		Ok(tex)
	}

	pub fn new_texture_from_iter<Px, I>(
		&mut self,
		iter: I,
		vk_fmt: Format,
		dimensions: [u32; 2],
		mip: u32,
	) -> Result<texture::Texture, GenericEngineError>
	where
		Px: Send + Sync + bytemuck::Pod,
		[Px]: vulkano::buffer::BufferContents,
		I: IntoIterator<Item = Px>,
		I::IntoIter: ExactSizeIterator,
	{
		let (tex, staging_work) = texture::Texture::new_from_iter(self.memory_allocator.clone(), iter, vk_fmt, dimensions, mip)?;
		self.submit_transfer(staging_work.into())?;
		Ok(tex)
	}

	/// Create an immutable buffer, initialized with `data` for `usage`.
	pub fn new_immutable_buffer_from_iter<I, T>(
		&mut self,
		data: I,
		buf_usage: BufferUsage,
	) -> Result<Subbuffer<[T]>, GenericEngineError>
	where
		T: Send + Sync + bytemuck::Pod,
		I: IntoIterator<Item = T>,
		I::IntoIter: ExactSizeIterator,
		[T]: vulkano::buffer::BufferContents,
	{
		let buffer_info = BufferCreateInfo {
			usage: BufferUsage::TRANSFER_SRC,
			..Default::default()
		};
		let staging_allocation_info = AllocationCreateInfo {
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buf = Buffer::from_iter(self.memory_allocator.clone(), buffer_info, staging_allocation_info, data)?;

		let buffer_info = BufferCreateInfo {
			sharing: Sharing::Concurrent(self.get_queue_families().into()),
			usage: buf_usage | BufferUsage::TRANSFER_DST,
			..Default::default()
		};
		let buf = Buffer::new_slice(self.memory_allocator.clone(), buffer_info, Default::default(), staging_buf.len())?;
		self.submit_transfer(CopyBufferInfo::buffers(staging_buf, buf.clone()).into())?;
		Ok(buf)
	}

	/// Create an immutable buffer, initialized with `data` for `usage`.
	pub fn new_immutable_buffer_from_data<T>(
		&mut self,
		data: T,
		buf_usage: BufferUsage,
	) -> Result<Subbuffer<T>, GenericEngineError>
	where
		T: vulkano::buffer::BufferContents,
	{
		let buffer_info = BufferCreateInfo {
			usage: BufferUsage::TRANSFER_SRC,
			..Default::default()
		};
		let staging_allocation_info = AllocationCreateInfo {
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buf = Buffer::from_data(self.memory_allocator.clone(), buffer_info, staging_allocation_info, data)?;

		let buffer_info = BufferCreateInfo {
			sharing: Sharing::Concurrent(self.get_queue_families().into()),
			usage: buf_usage | BufferUsage::TRANSFER_DST,
			..Default::default()
		};
		let buf = Buffer::new_sized(self.memory_allocator.clone(), buffer_info, Default::default())?;
		self.submit_transfer(CopyBufferInfo::buffers(staging_buf, buf.clone()).into())?;
		Ok(buf)
	}

	/// Create just a device-local buffer. It'll be initialized using the common staging buffer allocator,
	/// and it can be updated using that as well.
	pub fn new_staged_buffer_from_data<T>(
		&mut self,
		data: T,
		buf_usage: BufferUsage,
	) -> Result<Subbuffer<T>, GenericEngineError>
	where
		[T]: vulkano::buffer::BufferContents,
		T: Send + Sync + bytemuck::Pod,
	{
		let buffer_info = BufferCreateInfo {
			sharing: Sharing::Concurrent(self.get_queue_families().into()),
			usage: buf_usage | BufferUsage::TRANSFER_DST,
			..Default::default()
		};
		let gpu_buf = Buffer::new_sized(self.memory_allocator.clone(), buffer_info, Default::default())?;
		self.copy_to_buffer(data, gpu_buf.clone())?;
		Ok(gpu_buf)
	}

	/// Get a staging buffer from the common staging buffer allocator, and queue an upload using it.
	pub fn copy_to_buffer<T>(&mut self, data: T, dst_buf: Subbuffer<T>) -> Result<(), GenericEngineError>
	where
		[T]: vulkano::buffer::BufferContents,
		T: Send + Sync + bytemuck::Pod,
	{
		let staging_buf = self
			.staging_buffer_allocator
			.lock()
			.or_else(|_| Err("Staging buffer allocator mutex is poisoned!"))?
			.allocate_sized()?;
		*staging_buf.write()? = data;
		self.copy_buffer(staging_buf, dst_buf)?;
		Ok(())
	}

	fn get_queue_families(&self) -> Vec<u32>
	{
		self.graphics_queue.device().active_queue_family_indices().into()
	}

	pub fn new_descriptor_set(
		&self,
		pipeline_name: &str,
		set: usize,
		writes: impl IntoIterator<Item = WriteDescriptorSet>,
	) -> Result<Arc<PersistentDescriptorSet>, GenericEngineError>
	{
		self.material_pipelines
			.get(pipeline_name)
			.ok_or(PipelineNotLoaded)?
			.new_descriptor_set(set, writes)
	}

	/// Issue a new secondary command buffer builder to begin recording to.
	/// It will be set up for drawing to color and depth images with the given format,
	/// and with a viewport as large as `viewport_dimensions`.
	fn new_secondary_command_buffer(
		&self,
		color_attachment_formats: Vec<Option<Format>>,
		depth_attachment_format: Option<Format>,
		viewport_dimensions: [u32; 2]
	) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, Validated<VulkanError>>
	{
		let inherit_rp = CommandBufferInheritanceRenderPassType::BeginRendering(CommandBufferInheritanceRenderingInfo {
			color_attachment_formats,
			depth_attachment_format,
			..Default::default()
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
		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
			depth_range: 0.0..=1.0,
		};
		cb.set_viewport(0, [viewport].as_slice().into())?;

		Ok(cb)
	}
	pub fn record_main_draws(&self) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, Validated<VulkanError>>
	{
		self.new_secondary_command_buffer(
			vec![ Some(self.main_render_target.color_image().image().format()) ], 
			Some(self.main_render_target.depth_image().image().format()),
			self.swapchain_dimensions()
		)
	}

	/// Start recording commands for moment-based OIT. This will bind the pipeline for you, since it doesn't need to do
	/// anything specific to materials (it only reads the alpha channel of each texture).
	pub fn record_transparency_moments_draws(
		&self, 
		projview: Mat4,
	) -> Result<(AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, &pipeline::Pipeline), GenericEngineError>
	{
		let mut cb = self.new_secondary_command_buffer(
			vec![ 
				Some(Format::R32G32B32A32_SFLOAT),
				Some(Format::R32_SFLOAT),
				Some(Format::R32_SFLOAT),
			], 
			Some(self.main_render_target.depth_image().image().format()),
			self.swapchain_dimensions()
		)?;
		let pl = self.transparency_renderer.get_moments_pipeline();
		pl.bind(&mut cb)?;
		cb.push_constants(pl.layout(), 0, projview)?;
		Ok((cb, pl))
	}
	pub fn record_transparency_draws(
		&self,
		first_bind_pipeline: &pipeline::Pipeline,
		projview: Mat4,
	) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, GenericEngineError>
	{
		let color_formats = vec![ Some(Format::R16G16B16A16_SFLOAT), Some(Format::R8_UNORM) ];

		let mut cb = self.new_secondary_command_buffer(color_formats, Some(Format::D16_UNORM), self.swapchain_dimensions())?;
		first_bind_pipeline.bind_transparency(&mut cb)?;
		cb.bind_descriptor_sets(
			PipelineBindPoint::Graphics,
			first_bind_pipeline.layout_transparency()?, 
			2,
			vec![self.transparency_renderer.get_stage3_inputs()]
		)?;
		cb.push_constants(first_bind_pipeline.layout(), 0, projview)?;

		Ok(cb)
	}
	pub fn record_ui_draws(&self) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, Validated<VulkanError>>
	{
		self.new_secondary_command_buffer(
			vec![ Some(self.main_render_target.color_image().image().format()) ], 
			Some(Format::D16_UNORM), 
			self.swapchain_dimensions()
		)
	}

	/// Tell the swapchain to go to the next image.
	/// The image size *may* change here.
	/// This must only be called once per frame.
	///
	/// This returns the acquired swapchain image.
	fn next_swapchain_image(&mut self) -> Result<Arc<Image>, GenericEngineError>
	{
		let (image, dimensions_changed) = self.swapchain.get_next_image()?;
		if dimensions_changed {
			self.resize_everything_else()?;
		}
		self.resize_this_frame = dimensions_changed;

		Ok(image)
	}

	fn resize_everything_else(&mut self) -> Result<(), GenericEngineError>
	{
		// Update images to match the current window size.
		self.main_render_target
			.resize(self.memory_allocator.clone(), self.swapchain.dimensions())?;

		let swapchain_dim = self.swapchain.dimensions();
		let intermediate_img_create_info = ImageCreateInfo {
			format: Format::B8G8R8A8_SRGB,
			extent: [ swapchain_dim[0], swapchain_dim[1], 1 ],
			usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
			..Default::default()
		};
		self.intermediate_srgb_img = 
			Image::new(self.memory_allocator.clone(), intermediate_img_create_info, AllocationCreateInfo::default())?;

		self.transparency_renderer.resize_image(
			self.memory_allocator.clone(), 
			self.descriptor_set_allocator.clone(), 
			self.swapchain.dimensions()
		)?;

		Ok(())
	}

	pub fn resize_swapchain(&mut self) -> Result<(), GenericEngineError>
	{
		self.resize_this_frame = self.swapchain.fit_window()?;
		self.resize_everything_else()?;
		Ok(())
	}

	/// Create a command buffer for a transfer.
	/// TODO: should we wait for more work before we build a command buffer? it's probably inefficient to submit a command
	/// buffer with just a single command...
	fn create_staging_command_buffer(
		&self,
		work: StagingWork,
		queue_family: u32,
	) -> Result<Arc<PrimaryAutoCommandBuffer>, GenericEngineError>
	{
		let mut staging_cb_builder = AutoCommandBufferBuilder::primary(
			&self.command_buffer_allocator,
			queue_family,
			CommandBufferUsage::OneTimeSubmit,
		)?;
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
	pub fn copy_buffer(&mut self, src: Subbuffer<impl ?Sized>, dst: Subbuffer<impl ?Sized>) -> Result<(), GenericEngineError>
	{
		self.submit_transfer_on_graphics_queue(CopyBufferInfo::buffers(src, dst).into())
	}

	pub fn add_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		self.trm.lock().unwrap().add_cb(cb);
	}
	pub fn add_transparency_moments_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		self.transparency_renderer.add_transparency_moments_cb(cb);
	}
	pub fn add_transparency_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		self.transparency_renderer.add_transparency_cb(cb);
	}
	pub fn add_ui_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		self.trm.lock().unwrap().add_ui_cb(cb);
	}

	fn submit_commands(&mut self, built_cb: Arc<PrimaryAutoCommandBuffer>) -> Result<(), GenericEngineError>
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
		// finalize the rendering for this frame by executing the secondary command buffers
		let mut primary_cb_builder = AutoCommandBufferBuilder::primary(
			&self.command_buffer_allocator,
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
		)?;

		let main_render_info = RenderingInfo {
			color_attachments: vec![
				Some(RenderingAttachmentInfo {
					// `load_op` default `DontCare` is used since drawing the skybox effectively clears the image for us
					store_op: AttachmentStoreOp::Store,
					..RenderingAttachmentInfo::image_view(self.main_render_target.color_image().clone())
				}),
			],
			depth_attachment: Some(RenderingAttachmentInfo{
				// `load_op` is `DontCare` here too since the skybox clears it with 1.0
				store_op: AttachmentStoreOp::Store, // order-independent transparency needs this to be `Store`
				..RenderingAttachmentInfo::image_view(self.main_render_target.depth_image().clone())
			}),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		};
		let ui_render_info = RenderingInfo {
			color_attachments: vec![
				Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Load,
					store_op: AttachmentStoreOp::Store,
					..RenderingAttachmentInfo::image_view(self.main_render_target.color_image().clone())
				}),
			],
			depth_attachment: Some(RenderingAttachmentInfo{
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::DontCare, 
				..RenderingAttachmentInfo::image_view(self.main_render_target.depth_image().clone())
			}),
			contents: SubpassContents::SecondaryCommandBuffers,
			..Default::default()
		};

		let command_buffers = self.trm.lock().unwrap().take_built_command_buffers();
		primary_cb_builder
			.begin_rendering(main_render_info)?
			.execute_commands_from_vec(command_buffers)?
			.end_rendering()?;

		self.transparency_renderer.process_transparency(
			&mut primary_cb_builder, 
			self.main_render_target.color_image().clone(),
			self.main_render_target.depth_image().clone()
		)?;

		let ui_cb = self.trm.lock().unwrap().take_ui_cb();	
		let blit_info = BlitImageInfo::images(
			self.main_render_target.color_image().image().clone(),
			self.intermediate_srgb_img.clone(),
		);
		let copy_info = CopyImageInfo::images(self.intermediate_srgb_img.clone(), self.next_swapchain_image()?);
		primary_cb_builder
			.begin_rendering(ui_render_info)?
			.execute_commands_from_vec(ui_cb)?
			.end_rendering()?
			.blit_image(blit_info)? // convert from linear to non-linear sRGB
			.copy_image(copy_info)?; // copy non-linear image to swapchain image, the latter expected to be B8G8R8A8_UNORM
		self.submit_commands(primary_cb_builder.build()?)?;

		Ok(())
	}

	pub fn descriptor_set_allocator(&self) -> &Arc<StandardDescriptorSetAllocator>
	{
		&self.descriptor_set_allocator
	}
	pub fn memory_allocator(&self) -> &StandardMemoryAllocator
	{
		&self.memory_allocator
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

// Get and print the video modes supported by the given monitor, and return the monitor's current video mode.
// This may return `None` if the monitor is currently using a video mode that it really doesn't support,
// although that should be rare.
fn get_video_modes(mon: winit::monitor::MonitorHandle) -> Option<winit::monitor::VideoMode>
{
	let mon_name = mon.name().unwrap_or_else(|| "[no longer exists]".to_string());

	let current_video_mode = mon
		.video_modes()
		.find(|vm| vm.size() == mon.size() && vm.refresh_rate_millihertz() == mon.refresh_rate_millihertz().unwrap_or(0));

	let mut video_modes: Vec<_> = mon.video_modes().collect();
	log::info!("All video modes supported by current monitor (\"{mon_name}\"):");

	// filter the video modes to those with >=1280 width, >=720 height, and not vertical (width <= height)
	video_modes.retain(|video_mode| {
		// print unfiltered video modes while we're at it
		let video_mode_suffix = match &current_video_mode {
			Some(cur_vm) if video_mode == cur_vm  => " <- likely current primary monitor video mode",
			_ => "",
		};
		log::info!("{}{}", format_video_mode(video_mode), video_mode_suffix);

		let size = video_mode.size();
		size.width >= 1280 && size.height >= 720 && size.width >= size.height
	});

	// filter the video modes to the highest refresh rate for each size
	// (sort beforehand so that highest refresh rate for each size comes first,
	// then remove duplicates with the same size and bit depth)
	video_modes.sort();
	video_modes.dedup_by_key(|video_mode| (video_mode.size(), video_mode.bit_depth()));

	log::info!("Filtered video modes:");
	for vm in &video_modes {
		log::info!("{}", format_video_mode(vm))
	}

	//(video_modes, current_video_mode)
	current_video_mode
}
fn format_video_mode(video_mode: &winit::monitor::VideoMode) -> String
{
	let size = video_mode.size();
	let refresh_rate_hz = video_mode.refresh_rate_millihertz() / 1000;
	let refresh_rate_thousandths = video_mode.refresh_rate_millihertz() % 1000;
	format!(
		"{} x {} @ {}.{:0>3} Hz {}-bit",
		size.width,
		size.height,
		refresh_rate_hz,
		refresh_rate_thousandths,
		video_mode.bit_depth()
	)
}

/*/// Bind the given descriptor sets to the currently bound pipeline on the given command buffer builder.
/// This will fail if there is no pipeline currently bound.
pub fn bind_descriptor_set<L, S>(
	cb: &mut AutoCommandBufferBuilder<L>,
	bound_pipeline: Option<Arc<vulkano::pipeline::Pipeline>>,
	first_set: u32,
	descriptor_sets: S,
) -> Result<(), GenericEngineError>
where
	S: DescriptorSetsCollection,
{
	let pipeline_layout = bound_pipeline
		.ok_or("Attempted to bind descriptor set when no pipeline is bound!")?
		.layout()
		.clone();
	cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout, first_set, descriptor_sets);
	Ok(())
}

/// Push push constants to the currently bound pipeline on the given command buffer builder.
/// This will fail if there is no pipeline currently bound.
pub fn push_constants<L, Pc>(
	cb: &mut AutoCommandBufferBuilder<L>,
	bound_pipeline: Option<Arc<vulkano::pipeline::Pipeline>>,
	offset: u32, 
	data: Pc
) -> Result<(), GenericEngineError>
where
	Pc: BufferContents,
{
	let pipeline_layout = cb
		.state()
		.pipeline_graphics()
		.ok_or("Attempted to push constants when no pipeline is bound!")?
		.layout()
		.clone();
	cb.push_constants(pipeline_layout, offset, data);
	Ok(())
}*/

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

struct ThreadedRenderingManager
{
	built_command_buffers: Vec<Arc<dyn SecondaryCommandBufferAbstract>>,
	ui_cb: Vec<Arc<dyn SecondaryCommandBufferAbstract>>,
	default_capacity: usize,
}
impl ThreadedRenderingManager
{
	pub fn new(default_capacity: usize) -> Self
	{
		ThreadedRenderingManager {
			built_command_buffers: Vec::with_capacity(default_capacity),
			ui_cb: Vec::with_capacity(2),
			default_capacity,
		}
	}

	/// Add a secondary command buffer that has been built.
	pub fn add_cb(&mut self, command_buffer: Arc<SecondaryAutoCommandBuffer>)
	{
		self.built_command_buffers.push(command_buffer);
	}

	pub fn add_ui_cb(&mut self, command_buffer: Arc<SecondaryAutoCommandBuffer>)
	{
		self.ui_cb.push(command_buffer)
	}

	/// Take all of the secondary command buffers that have been built.
	pub fn take_built_command_buffers(&mut self) -> Vec<Arc<dyn SecondaryCommandBufferAbstract>>
	{
		std::mem::replace(&mut self.built_command_buffers, Vec::with_capacity(self.default_capacity))
	}
	pub fn take_ui_cb(&mut self) -> Vec<Arc<dyn SecondaryCommandBufferAbstract>>
	{
		std::mem::replace(&mut self.ui_cb, Vec::with_capacity(2))
	}
}

struct RenderTarget
{
	// An FP16, linear gamma image which everything will be rendered to.
	color_image: Arc<ImageView>,
	depth_image: Arc<ImageView>,
}
impl RenderTarget
{
	pub fn new(memory_allocator: Arc<StandardMemoryAllocator>, dimensions: [u32; 2]) -> Result<Self, GenericEngineError>
	{
		let color_create_info = ImageCreateInfo {
			format: Format::R16G16B16A16_SFLOAT,
			extent: [ dimensions[0], dimensions[1], 1 ],
			usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
			..Default::default()
		};
		let color_image = Image::new(memory_allocator.clone(), color_create_info, AllocationCreateInfo::default())?;

		// NOTE: 24-bit depth formats are unsupported on a significant number of GPUs
		let depth_create_info = ImageCreateInfo {
			format: Format::D16_UNORM,
			extent: [ dimensions[0], dimensions[1], 1 ],
			usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
			..Default::default()
		};
		let depth_image = Image::new(memory_allocator, depth_create_info, AllocationCreateInfo::default())?;

		Ok(Self {
			color_image: ImageView::new_default(color_image)?,
			depth_image: ImageView::new_default(depth_image)?,
		})
	}

	pub fn resize(&mut self, memory_allocator: Arc<StandardMemoryAllocator>, dimensions: [u32; 2])
		-> Result<(), GenericEngineError>
	{
		let color_info = ImageCreateInfo {
			extent: [ dimensions[0], dimensions[1], 1],
			format: Format::R16G16B16A16_SFLOAT,
			usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
			..Default::default()
		};
		self.color_image = ImageView::new_default(Image::new(
			memory_allocator.clone(), 
			color_info, 
			AllocationCreateInfo::default()
		)?)?;

		let depth_info = ImageCreateInfo {
			extent: [ dimensions[0], dimensions[1], 1],
			format: Format::D16_UNORM,
			usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
			..Default::default()
		};
		self.depth_image = ImageView::new_default(Image::new(memory_allocator, depth_info, AllocationCreateInfo::default())?)?;
		Ok(())
	}

	pub fn color_image(&self) -> &Arc<ImageView>
	{
		&self.color_image
	}
	pub fn depth_image(&self) -> &Arc<ImageView>
	{
		&self.depth_image
	}
}
