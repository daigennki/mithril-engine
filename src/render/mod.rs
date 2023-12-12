/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod model;
pub mod pipeline;
mod render_target;
pub mod skybox;
mod swapchain;
pub mod texture;
pub mod transparency;
mod vulkan_init;
pub mod workload;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use glam::*;

use vulkano::buffer::{
	allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
	subbuffer::Subbuffer,
	Buffer, BufferContents, BufferCreateInfo, BufferUsage,
};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderingInfo,
	CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
	RenderingAttachmentInfo, RenderingInfo, SecondaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::{
	allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	DescriptorSet,
};
use vulkano::device::{Device, Queue};
use vulkano::format::{ClearValue, Format};
use vulkano::image::{
	sampler::{Filter, Sampler, SamplerCreateInfo},
	view::ImageView,
};
use vulkano::memory::{
	allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
	MemoryPropertyFlags,
};
use vulkano::pipeline::graphics::{
	depth_stencil::CompareOp,
	subpass::{PipelineRenderingCreateInfo, PipelineSubpassType},
	viewport::Viewport,
	GraphicsPipeline,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;
use vulkano::sync::{
	future::{FenceSignalFuture, NowFuture},
	GpuFuture,
};
use vulkano::{DeviceSize, Validated, VulkanError};

use crate::material::Material;
use crate::GenericEngineError;
use texture::Texture;

// Format used for main depth buffer.
// NOTE: While [NVIDIA recommends](https://developer.nvidia.com/blog/vulkan-dos-donts/) using a 24-bit depth format
// (`D24_UNORM_S8_UINT`), it doesn't seem to be very well-supported outside of NVIDIA GPUs. Only about 70% of GPUs on Windows
// and 50% of GPUs on Linux seem to support it, while `D16_UNORM` and `D32_SFLOAT` both have 100% support.
// (https://vulkan.gpuinfo.org/listoptimaltilingformats.php)
pub const MAIN_DEPTH_FORMAT: Format = Format::D16_UNORM;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	device: Arc<Device>,
	graphics_queue: Arc<Queue>,
	swapchain: swapchain::Swapchain,
	main_render_target: render_target::RenderTarget,
	descriptor_set_allocator: StandardDescriptorSetAllocator,
	memory_allocator: Arc<StandardMemoryAllocator>,
	command_buffer_allocator: StandardCommandBufferAllocator,

	cb_3d: Mutex<VecDeque<Arc<SecondaryAutoCommandBuffer>>>,

	transparency_renderer: transparency::MomentTransparencyRenderer,

	light_set_layout: Arc<DescriptorSetLayout>,

	// User-accessible material pipelines. Optional transparency pipeline may also be specified.
	material_pipelines: HashMap<String, (Arc<GraphicsPipeline>, Option<Arc<GraphicsPipeline>>)>,

	// Loaded textures, with the key being the path relative to the current working directory
	textures: HashMap<PathBuf, Arc<Texture>>,

	allow_direct_buffer_access: bool,

	// Transfers to initialize buffers and images. If there is an asynchronous transfer queue,
	// these will be performed while the CPU is busy with building the draw command buffers.
	// Otherwise, these will be run at the beginning of the next graphics submission, before draws are performed.
	async_transfers: Vec<StagingWork>,
	transfer_queue: Option<Arc<Queue>>, // if there is a separate (preferably dedicated) transfer queue, use it for transfers
	transfer_future: Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>,

	// Buffer updates to run at the beginning of the next graphics submission.
	buffer_updates: Vec<Box<dyn UpdateBufferDataTrait>>,
	staging_buffer_allocator: Mutex<SubbufferAllocator>, // Used for the buffer updates.
	staging_buf_max_size: DeviceSize,                    // Maximum staging buffer usage for the entire duration of the program.
	staging_buf_usage_frame: DeviceSize,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) -> Result<Self, GenericEngineError>
	{
		let (graphics_queue, transfer_queue, allow_direct_buffer_access) = vulkan_init::vulkan_setup(game_name, event_loop)?;
		let vk_dev = graphics_queue.device().clone();

		let swapchain = swapchain::Swapchain::new(vk_dev.clone(), event_loop, game_name)?;

		let descriptor_set_allocator =
			StandardDescriptorSetAllocator::new(vk_dev.clone(), StandardDescriptorSetAllocatorCreateInfo::default());
		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_dev.clone()));

		// The counts below are multiplied by the number of swapchain images, to account for previous submissions.
		// - Primary: One for graphics, another for async transfers, each on separate queue families.
		// - Secondary: Only up to two should be created per thread.
		let cb_alloc_info = StandardCommandBufferAllocatorCreateInfo {
			primary_buffer_count: swapchain.image_count(),
			secondary_buffer_count: 2 * swapchain.image_count(),
			..Default::default()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(vk_dev.clone(), cb_alloc_info);

		let main_render_target = render_target::RenderTarget::new(
			memory_allocator.clone(),
			&descriptor_set_allocator,
			swapchain.dimensions(),
			swapchain.format(),
			swapchain.color_space(),
		)?;
		let transparency_renderer = transparency::MomentTransparencyRenderer::new(
			memory_allocator.clone(),
			&descriptor_set_allocator,
			swapchain.dimensions(),
		)?;

		/* descriptor set with everything lighting- and shadow-related */
		let shadow_sampler_info = SamplerCreateInfo {
			mag_filter: Filter::Linear,
			min_filter: Filter::Linear,
			compare: Some(CompareOp::LessOrEqual),
			..Default::default()
		};
		let shadow_sampler = Sampler::new(vk_dev.clone(), shadow_sampler_info)?;
		let light_bindings = [
			DescriptorSetLayoutBinding {
				// binding 0: shadow sampler
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![shadow_sampler],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
			},
			DescriptorSetLayoutBinding {
				// binding 1: directional light buffer
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
			},
			DescriptorSetLayoutBinding {
				// binding 2: directional light shadow
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
			},
		];
		let light_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: (0..).zip(light_bindings).collect(),
			..Default::default()
		};
		let light_set_layout = DescriptorSetLayout::new(vk_dev.clone(), light_set_layout_info)?;

		let pool_create_info = SubbufferAllocatorCreateInfo {
			arena_size: 8 * 1024 * 1024, // this should be adjusted based on actual memory usage
			buffer_usage: BufferUsage::TRANSFER_SRC,
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buffer_allocator = Mutex::new(SubbufferAllocator::new(memory_allocator.clone(), pool_create_info));

		// the capacity of these should be adjusted based on number of transfers that might be done in one frame
		let buffer_updates = Vec::with_capacity(16);
		let async_transfers = Vec::with_capacity(200);

		Ok(RenderContext {
			device: vk_dev,
			swapchain,
			graphics_queue,
			descriptor_set_allocator,
			memory_allocator,
			command_buffer_allocator,
			cb_3d: Mutex::new(VecDeque::with_capacity(2)),
			material_pipelines: HashMap::new(),
			main_render_target,
			transparency_renderer,
			light_set_layout,
			textures: HashMap::new(),
			allow_direct_buffer_access,
			async_transfers,
			transfer_queue,
			transfer_future: None,
			buffer_updates,
			staging_buffer_allocator,
			staging_buf_max_size: 0,
			staging_buf_usage_frame: 0,
		})
	}

	/// Load a material shader pipeline into memory.
	/// Returns a set layout representing the descriptor set interface unique to the material.
	pub fn load_material_pipeline(&mut self, mat: &dyn Material) -> Result<Arc<DescriptorSetLayout>, GenericEngineError>
	{
		let name = mat.material_name();
		let transparency_set_layout = self.transparency_renderer.get_stage3_inputs().layout().clone();
		let (pipeline, transparency_pipeline, mat_set_layout) =
			mat.load_pipeline(self.light_set_layout.clone(), transparency_set_layout)?;

		self.material_pipelines
			.insert(name.to_string(), (pipeline, transparency_pipeline));

		Ok(mat_set_layout)
	}
	pub fn get_pipeline(&self, name: &str) -> Option<&Arc<GraphicsPipeline>>
	{
		self.material_pipelines.get(name).map(|(pl, _)| pl)
	}
	pub fn get_transparency_pipeline(&self, name: &str) -> Option<&Arc<GraphicsPipeline>>
	{
		self.material_pipelines.get(name).and_then(|(_, pl)| pl.as_ref())
	}

	pub fn get_light_set_layout(&self) -> &Arc<DescriptorSetLayout>
	{
		&self.light_set_layout
	}

	/// Load an image file as a texture into memory.
	/// If the image was already loaded, it'll use the corresponding texture.
	pub fn get_texture(&mut self, path: &Path) -> Result<Arc<Texture>, GenericEngineError>
	{
		match self.textures.get(path) {
			Some(tex) => Ok(tex.clone()),
			None => {
				let (tex, staging_work) = texture::Texture::new(
					self.memory_allocator.clone(),
					&mut self.staging_buffer_allocator.lock().unwrap(),
					path,
				)?;
				self.add_transfer(staging_work.into());
				let tex_arc = Arc::new(tex);
				self.textures.insert(path.to_path_buf(), tex_arc.clone());
				Ok(tex_arc)
			}
		}
	}

	pub fn new_cubemap_texture(&mut self, faces: [PathBuf; 6]) -> Result<texture::CubemapTexture, GenericEngineError>
	{
		let (tex, staging_work) = texture::CubemapTexture::new(
			self.memory_allocator.clone(),
			&mut self.staging_buffer_allocator.lock().unwrap(),
			faces,
		)?;
		self.add_transfer(staging_work.into());
		Ok(tex)
	}

	pub fn new_texture_from_slice<Px>(
		&mut self,
		data: &[Px],
		vk_fmt: Format,
		dimensions: [u32; 2],
		mip: u32,
		array_layers: u32,
	) -> Result<texture::Texture, GenericEngineError>
	where
		Px: BufferContents + Copy,
	{
		let (tex, staging_work) = texture::Texture::new_from_slice(
			self.memory_allocator.clone(),
			&mut self.staging_buffer_allocator.lock().unwrap(),
			data,
			vk_fmt,
			dimensions,
			mip,
			array_layers,
		)?;
		self.add_transfer(staging_work.into());
		Ok(tex)
	}

	/// Create a device-local buffer from a slice, initialized with `data` for `usage`.
	/// For stuff that isn't an array, just put the data into a single-element slice, like `[data]`.
	pub fn new_buffer<T>(&mut self, data: &[T], usage: BufferUsage) -> Result<Subbuffer<[T]>, GenericEngineError>
	where
		T: BufferContents + Copy,
	{
		let data_len = data.len().try_into()?;
		let data_size_bytes = data.len() * std::mem::size_of::<T>();
		let buf;
		if self.allow_direct_buffer_access {
			log::debug!("Allocating direct buffer of {} bytes", data_size_bytes);
			// When possible, upload directly to the new buffer memory.
			let buf_info = BufferCreateInfo {
				usage,
				..Default::default()
			};
			let alloc_info = AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter {
					required_flags: MemoryPropertyFlags::HOST_VISIBLE,
					preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL,
					not_preferred_flags: MemoryPropertyFlags::HOST_CACHED
						| MemoryPropertyFlags::DEVICE_COHERENT
						| MemoryPropertyFlags::DEVICE_UNCACHED,
				},
				..Default::default()
			};
			buf = Buffer::new_slice(self.memory_allocator.clone(), buf_info, alloc_info, data_len)?;
			buf.write().unwrap().copy_from_slice(data);
		} else {
			log::debug!("Allocating buffer of {} bytes", data_size_bytes);
			// If direct uploads aren't possible, create a staging buffer on the CPU side,
			// then submit a transfer command to the new buffer on the GPU side.
			let staging_buf = self
				.staging_buffer_allocator
				.lock()
				.unwrap()
				.allocate_slice(data.len().try_into()?)?;
			staging_buf.write().unwrap().copy_from_slice(data);

			let buf_info = BufferCreateInfo {
				usage: usage | BufferUsage::TRANSFER_DST,
				..Default::default()
			};
			buf = Buffer::new_slice(
				self.memory_allocator.clone(),
				buf_info,
				AllocationCreateInfo::default(),
				data_len,
			)?;

			self.add_transfer(CopyBufferInfo::buffers(staging_buf, buf.clone()).into());
		}
		Ok(buf)
	}

	/// Update a buffer at the begninning of the next graphics submission.
	pub fn update_buffer<T>(&mut self, data: &[T], dst_buf: Subbuffer<[T]>) -> Result<(), GenericEngineError>
	where
		T: BufferContents + Copy,
	{
		// This will be submitted to the graphics queue since we're copying to an existing buffer,
		// which might be in use by a previous submission.
		if self.buffer_updates.len() == self.buffer_updates.capacity() {
			log::warn!(
				"Re-allocating `Vec` for buffer updates to {}! Consider increasing its initial capacity.",
				self.buffer_updates.len() + 1
			);
		}
		self.buffer_updates.push(Box::new(UpdateBufferData {
			dst_buf,
			data: data.into(),
		}));

		Ok(())
	}

	/// Add staging work for new objects.
	/// If there is a separate transfer queue, this will be performed asynchronously with the CPU
	/// building the draw command buffers.
	fn add_transfer(&mut self, work: StagingWork)
	{
		self.staging_buf_usage_frame += work.buf_size();
		if self.async_transfers.len() == self.async_transfers.capacity() {
			log::warn!(
				"Re-allocating `Vec` for asynchronous transfers to {}! Consider increasing its initial capacity.",
				self.async_transfers.len() + 1
			);
		}
		self.async_transfers.push(work);
	}

	/// Submit the asynchronous transfers that are waiting.
	/// Run this just before beginning to build the draw command buffers,
	/// so that the transfers can be done while the CPU is busy with building the draw command buffers.
	///
	/// This does nothing if there is no asynchronous transfer queue. In such a case, the transfers will
	/// instead be done at the beginning of the graphics submission on the graphics queue.
	pub fn submit_async_transfers(&mut self) -> Result<(), GenericEngineError>
	{
		if let Some(q) = self.transfer_queue.as_ref() {
			if self.async_transfers.len() > 0 {
				let mut cb = AutoCommandBufferBuilder::primary(
					&self.command_buffer_allocator,
					q.queue_family_index(),
					CommandBufferUsage::OneTimeSubmit,
				)?;

				for work in self.async_transfers.drain(..) {
					work.add_command(&mut cb)?;
				}

				let transfer_future = cb.build()?.execute(q.clone())?.then_signal_fence_and_flush()?;

				// This panics here if there's an unused future, because it *must* have been used when
				// the draw commands were submitted last frame. Otherwise, we can't guarantee that transfers
				// have finished by the time the draws that need them are performed.
				assert!(self.transfer_future.replace(transfer_future).is_none());
			}
		}

		Ok(())
	}

	/// Issue a new secondary command buffer builder to begin recording to. It will be set up for
	/// drawing to the subpass that the given pipeline was set up with, will initially bind that
	/// same pipeline, and will initially have a viewport as large as`viewport_extent`.
	pub fn gather_commands(
		&self,
		first_pipeline: Arc<GraphicsPipeline>,
		viewport_extent: [u32; 2],
	) -> Result<AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>, Validated<VulkanError>>
	{
		let PipelineRenderingCreateInfo {
			color_attachment_formats,
			depth_attachment_format,
			stencil_attachment_format,
			..
		} = match first_pipeline.subpass() {
			PipelineSubpassType::BeginRenderPass(_) => unimplemented!(),
			PipelineSubpassType::BeginRendering(r) => r.clone(),
		};
		let rendering_inheritance = CommandBufferInheritanceRenderingInfo {
			color_attachment_formats,
			depth_attachment_format,
			stencil_attachment_format,
			..Default::default()
		};

		let mut cb = AutoCommandBufferBuilder::secondary(
			&self.command_buffer_allocator,
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
			CommandBufferInheritanceInfo {
				render_pass: Some(rendering_inheritance.into()),
				..Default::default()
			},
		)?;

		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: [viewport_extent[0] as f32, viewport_extent[1] as f32],
			depth_range: 0.0..=1.0,
		};
		cb.set_viewport(0, [viewport].as_slice().into())?;
		cb.bind_pipeline_graphics(first_pipeline.clone())?;

		Ok(cb)
	}

	fn resize_everything_else(&mut self) -> Result<(), GenericEngineError>
	{
		// Update images to match the current swapchain image extent.
		self.main_render_target = render_target::RenderTarget::new(
			self.memory_allocator.clone(),
			&self.descriptor_set_allocator,
			self.swapchain.dimensions(),
			self.swapchain.format(),
			self.swapchain.color_space(),
		)?;
		self.transparency_renderer.resize_image(
			self.memory_allocator.clone(),
			&self.descriptor_set_allocator,
			self.swapchain.dimensions(),
		)?;

		Ok(())
	}

	pub fn add_cb(&self, cb: Arc<SecondaryAutoCommandBuffer>)
	{
		self.cb_3d.lock().unwrap().push_back(cb);
	}

	/// Submit all the command buffers for this frame to actually render them to the image.
	pub fn submit_frame(
		&mut self,
		ui_cb: Option<Arc<SecondaryAutoCommandBuffer>>,
		dir_light_shadows: Vec<(Arc<SecondaryAutoCommandBuffer>, Arc<ImageView>)>,
	) -> Result<(), GenericEngineError>
	{
		let mut primary_cb_builder = AutoCommandBufferBuilder::primary(
			&self.command_buffer_allocator,
			self.graphics_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
		)?;

		// buffer updates
		if self.buffer_updates.len() > 0 {
			let mut staging_buf_alloc_guard = self.staging_buffer_allocator.lock().unwrap();

			for buf_update in self.buffer_updates.drain(..) {
				buf_update.add_command(&mut primary_cb_builder, &mut staging_buf_alloc_guard)?;
				self.staging_buf_usage_frame += buf_update.data_size();
			}

			// gather stats on staging buffer usage
			if self.staging_buf_usage_frame > self.staging_buf_max_size {
				self.staging_buf_max_size = self.staging_buf_usage_frame;
				log::debug!("max staging buffer usage per frame: {} bytes", self.staging_buf_max_size);
			}

			self.staging_buf_usage_frame = 0;
		}

		// do async transfers that couldn't be submitted earlier
		for work in self.async_transfers.drain(..) {
			work.add_command(&mut primary_cb_builder)?;
		}

		let transfer_future = self.transfer_future.take();

		// Sometimes no image may be returned because the image is out of date or the window is minimized,
		// in which case, don't present.
		if let Some(swapchain_image_view) = self.swapchain.get_next_image()? {
			if self.swapchain.extent_changed() {
				self.resize_everything_else()?;
			}

			// shadows
			for (shadow_cb, shadow_layer_image_view) in dir_light_shadows {
				let shadow_render_info = RenderingInfo {
					depth_attachment: Some(RenderingAttachmentInfo {
						load_op: AttachmentLoadOp::Clear,
						store_op: AttachmentStoreOp::Store,
						clear_value: Some(ClearValue::Depth(1.0)),
						..RenderingAttachmentInfo::image_view(shadow_layer_image_view)
					}),
					contents: SubpassContents::SecondaryCommandBuffers,
					..Default::default()
				};
				primary_cb_builder
					.begin_rendering(shadow_render_info)?
					.execute_commands(shadow_cb)?
					.end_rendering()?;
			}

			// skybox (effectively clears the image)
			let sky_render_info = RenderingInfo {
				color_attachments: vec![Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::DontCare,
					store_op: AttachmentStoreOp::Store,
					..RenderingAttachmentInfo::image_view(self.main_render_target.color_image().clone())
				})],
				contents: SubpassContents::SecondaryCommandBuffers,
				..Default::default()
			};
			primary_cb_builder
				.begin_rendering(sky_render_info)?
				.execute_commands(self.cb_3d.lock().unwrap().pop_front().unwrap())?
				.end_rendering()?;

			// 3D
			let main_render_info = RenderingInfo {
				color_attachments: vec![Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Load,
					store_op: AttachmentStoreOp::Store,
					..RenderingAttachmentInfo::image_view(self.main_render_target.color_image().clone())
				})],
				depth_attachment: Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store, // order-independent transparency needs this to be `Store`
					clear_value: Some(ClearValue::Depth(1.0)),
					..RenderingAttachmentInfo::image_view(self.main_render_target.depth_image().clone())
				}),
				contents: SubpassContents::SecondaryCommandBuffers,
				..Default::default()
			};
			primary_cb_builder
				.begin_rendering(main_render_info)?
				.execute_commands(self.cb_3d.lock().unwrap().pop_front().unwrap())?
				.end_rendering()?;

			// 3D OIT
			self.transparency_renderer.process_transparency(
				&mut primary_cb_builder,
				self.main_render_target.color_image().clone(),
				self.main_render_target.depth_image().clone(),
			)?;

			// UI
			if let Some(some_ui_cb) = ui_cb {
				let ui_render_info = RenderingInfo {
					color_attachments: vec![Some(RenderingAttachmentInfo {
						load_op: AttachmentLoadOp::Load,
						store_op: AttachmentStoreOp::Store,
						..RenderingAttachmentInfo::image_view(self.main_render_target.color_image().clone())
					})],
					contents: SubpassContents::SecondaryCommandBuffers,
					..Default::default()
				};
				primary_cb_builder
					.begin_rendering(ui_render_info)?
					.execute_commands(some_ui_cb)?
					.end_rendering()?;
			}

			// blit the image to the swapchain image, converting it to the swapchain's color space if necessary
			self.main_render_target
				.blit_to_swapchain(&mut primary_cb_builder, swapchain_image_view)?;
		}

		// submit the built command buffer, presenting it if possible
		self.swapchain
			.submit(primary_cb_builder.build()?, self.graphics_queue.clone(), transfer_future)?;

		Ok(())
	}

	pub fn descriptor_set_allocator(&self) -> &StandardDescriptorSetAllocator
	{
		&self.descriptor_set_allocator
	}

	pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator>
	{
		&self.memory_allocator
	}

	pub fn device(&self) -> &Arc<Device>
	{
		&self.device
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

	pub fn swapchain_dimensions(&self) -> [u32; 2]
	{
		self.swapchain.dimensions()
	}

	pub fn get_transparency_renderer(&self) -> &transparency::MomentTransparencyRenderer
	{
		&self.transparency_renderer
	}

	/// Get the delta time for last frame.
	pub fn delta(&self) -> std::time::Duration
	{
		self.swapchain.delta()
	}
}

struct UpdateBufferData<T: BufferContents + Copy>
{
	pub dst_buf: Subbuffer<[T]>,
	pub data: Vec<T>,
}
impl<T: BufferContents + Copy> UpdateBufferDataTrait for UpdateBufferData<T>
{
	fn data_size(&self) -> DeviceSize
	{
		self.dst_buf.size()
	}

	fn add_command(
		&self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		subbuffer_allocator: &mut SubbufferAllocator,
	) -> Result<(), GenericEngineError>
	{
		let staging_buf = subbuffer_allocator.allocate_slice(self.data.len().try_into()?)?;
		staging_buf.write()?.copy_from_slice(self.data.as_slice());

		// TODO: actually use `update_buffer` when the `'static` requirement gets removed for the data
		cb_builder.copy_buffer(CopyBufferInfo::buffers(staging_buf, self.dst_buf.clone()))?;

		Ok(())
	}
}
trait UpdateBufferDataTrait: Send + Sync
{
	fn data_size(&self) -> DeviceSize;

	fn add_command(
		&self,
		_: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		_: &mut SubbufferAllocator,
	) -> Result<(), GenericEngineError>;
}

enum StagingWork
{
	CopyBuffer(CopyBufferInfo),
	CopyBufferToImage(CopyBufferToImageInfo),
}
impl StagingWork
{
	fn add_command(self, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
		-> Result<(), GenericEngineError>
	{
		match self {
			StagingWork::CopyBuffer(info) => cb_builder.copy_buffer(info)?,
			StagingWork::CopyBufferToImage(info) => cb_builder.copy_buffer_to_image(info)?,
		};
		Ok(())
	}

	fn buf_size(&self) -> DeviceSize
	{
		match self {
			StagingWork::CopyBuffer(info) => info.src_buffer.size(),
			StagingWork::CopyBufferToImage(info) => info.src_buffer.size(),
		}
	}
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
