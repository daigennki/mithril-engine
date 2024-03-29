/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
pub mod lighting;
pub mod model;
mod smaa;
pub mod ui;
mod wboit;
mod window;

use ddsfile::DxgiFormat;
use glam::*;
use shipyard::{UniqueView, UniqueViewMut};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{BufReader, Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::*;
use vulkano::command_buffer::{allocator::*, *};
use vulkano::descriptor_set::{allocator::*, layout::*, *};
use vulkano::device::{
	physical::{PhysicalDevice, PhysicalDeviceType},
	Device, DeviceOwned,
};
use vulkano::format::{Format, FormatFeatures};
use vulkano::image::{sampler::*, view::*, *};
use vulkano::memory::{allocator::*, DeviceAlignment, MemoryPropertyFlags};
use vulkano::pipeline::graphics::{
	color_blend::ColorBlendState, multisample::MultisampleState, subpass::PipelineRenderingCreateInfo, viewport::Viewport,
	GraphicsPipelineCreateInfo,
};
use vulkano::pipeline::{compute::ComputePipelineCreateInfo, layout::*, *};
use vulkano::render_pass::AttachmentStoreOp;
use vulkano::shader::ShaderStages;
use vulkano::swapchain::ColorSpace;
use vulkano::sync::future::{FenceSignalFuture, GpuFuture};
use vulkano::{DeviceSize, Version};
use winit::{event::WindowEvent, event_loop::EventLoop};

use crate::component::camera::CameraManager;
use crate::EngineError;
use lighting::LightManager;
use model::MeshManager;
use ui::Canvas;

/// Main depth/stencil image formats, in order from most to least preferred. Vulkan spec requires
/// either `D24_UNORM_S8_UINT` or `D32_SFLOAT_S8_UINT` to be supported.
const DEPTH_STENCIL_FORMAT_CANDIDATES: [Format; 3] = [
	Format::D24_UNORM_S8_UINT, // the most efficient in terms of overall performance, if available
	Format::D16_UNORM_S8_UINT, // AMD only
	Format::D32_SFLOAT_S8_UINT,
];

/// The arena size for buffer/image transfers. A single transfer must be no larger than this.
const STAGING_ARENA_SIZE: DeviceSize = 32 * 1024 * 1024;

/// The embedded texture file used when other texture files couldn't be loaded.
const ERROR_TEXTURE_BYTES: &[u8] = include_bytes!("error.dds");

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	window: window::GameWindow,
	memory_allocator: Arc<StandardMemoryAllocator>,
	command_buffer_allocator: StandardCommandBufferAllocator,
	single_set_allocator: StandardDescriptorSetAllocator,

	// Transfers to initialize buffers and images.
	// If this or the staging buffer arena gets full, the transfers will get submitted immediately.
	transfers: Vec<StagingWork>,
	staging_arenas: [Subbuffer<[u8]>; 2],
	current_arena: usize,
	staging_layout: Option<DeviceLayout>, // Current layout inside the current arena.
	transfer_future: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,

	skybox_pipeline: Arc<GraphicsPipeline>,
	skybox_tex_set: Option<Arc<PersistentDescriptorSet>>,

	transparency_renderer: wboit::WboitRenderer,
	//transparency_renderer: Option<moment_transparency::MomentTransparencyRenderer>,

	// Things related to the main color/depth/stencil images and gamma correction.
	depth_stencil_format: Format,
	depth_stencil_image: Option<Arc<ImageView>>,
	color_image: Option<Arc<ImageView>>,
	aa_mode: AntiAliasingMode,
	smaa_renderer: Option<smaa::SmaaRenderer>,
	aa_output_image: Option<Arc<ImageView>>,
	gamma_input_set: Option<Arc<PersistentDescriptorSet>>,
	gamma_pipeline: Arc<ComputePipeline>,

	// Loaded textures, with the key being the path relative to the current working directory.
	textures: HashMap<PathBuf, Arc<ImageView>>,
	error_texture: Option<Arc<ImageView>>, // fallback texture to use when errors occur in texture loading
}
impl RenderContext
{
	pub fn new(game_name: &str, app_version: Version, event_loop: &EventLoop<()>) -> crate::Result<Self>
	{
		let window = window::GameWindow::new(event_loop, game_name, app_version)?;
		let vk_dev = window.graphics_queue().device().clone();
		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_dev.clone()));

		// - Primary command buffers: One for each graphics submission on each thread, plus four for transfers.
		// - Secondary command buffers: Only one should be created per thread.
		let cb_alloc_info = StandardCommandBufferAllocatorCreateInfo {
			primary_buffer_count: window.image_count() + 4,
			secondary_buffer_count: window.image_count() + 1,
			..Default::default()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(vk_dev.clone(), cb_alloc_info);

		let staging_buf_info = BufferCreateInfo {
			usage: BufferUsage::TRANSFER_SRC,
			..Default::default()
		};
		let staging_alloc_info = AllocationCreateInfo {
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
			..Default::default()
		};
		let total_arena_size = STAGING_ARENA_SIZE * 2;
		let staging_buf = Buffer::new_slice(
			memory_allocator.clone(),
			staging_buf_info,
			staging_alloc_info,
			total_arena_size,
		)?;

		let single_set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 2,
			..Default::default()
		};
		let single_set_allocator = StandardDescriptorSetAllocator::new(vk_dev.clone(), single_set_alloc_info);

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
			.unwrap(); // unwrap since at least one of the formats must be supported

		let aa_mode_regex = regex::Regex::new("--aa_mode=(?<value>\\w+)").unwrap();
		let aa_mode = std::env::args()
			.collect::<Vec<_>>()
			.iter()
			.find_map(|arg| aa_mode_regex.captures(arg))
			.and_then(|caps| caps.name("value"))
			.map_or(AntiAliasingMode::Off, |value| match value.as_str() {
				"Multisample2" => AntiAliasingMode::Multisample2,
				"Multisample4" => AntiAliasingMode::Multisample4,
				"Smaa" | "SMAA" => AntiAliasingMode::Smaa,
				_ => AntiAliasingMode::Off,
			});
		let rasterization_samples = aa_mode.sample_count();

		let sky_pipeline_layout = {
			let cubemap_sampler = Sampler::new(vk_dev.clone(), SamplerCreateInfo::simple_repeat_linear_no_mipmap())?;
			let tex_binding = DescriptorSetLayoutBinding {
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![cubemap_sampler],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
			};
			let set_layout_info = DescriptorSetLayoutCreateInfo {
				bindings: [(0, tex_binding)].into(),
				..Default::default()
			};
			let set_layout = DescriptorSetLayout::new(vk_dev.clone(), set_layout_info)?;

			let pipeline_layout_info = PipelineLayoutCreateInfo {
				set_layouts: vec![set_layout.clone()],
				push_constant_ranges: vec![PushConstantRange {
					stages: ShaderStages::VERTEX,
					offset: 0,
					size: std::mem::size_of::<Mat4>().try_into().unwrap(),
				}],
				..Default::default()
			};
			PipelineLayout::new(vk_dev.clone(), pipeline_layout_info)?
		};

		let mut new_self = Self {
			window,
			memory_allocator,
			command_buffer_allocator,
			single_set_allocator,
			transfers: Vec::with_capacity(256),
			staging_arenas: staging_buf.split_at(STAGING_ARENA_SIZE).into(),
			current_arena: 0,
			staging_layout: None,
			transfer_future: None,
			skybox_pipeline: create_sky_pipeline(sky_pipeline_layout, rasterization_samples)?,
			skybox_tex_set: None,
			transparency_renderer: wboit::WboitRenderer::new(vk_dev.clone())?,
			aa_mode,
			smaa_renderer: None,
			depth_stencil_format,
			depth_stencil_image: None,
			color_image: None,
			aa_output_image: None,
			gamma_input_set: None,
			gamma_pipeline: create_gamma_pipeline(vk_dev.clone())?,
			textures: HashMap::new(),
			error_texture: None,
		};

		{
			let (format, extent, mip_levels, img_raw) = TextureSource::Bytes(ERROR_TEXTURE_BYTES).load().unwrap();
			let image_info = ImageCreateInfo {
				format,
				extent: [extent[0], extent[1], 1],
				mip_levels,
				usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
				..Default::default()
			};
			let image = new_self.new_image(&img_raw, image_info)?;
			new_self.error_texture = Some(ImageView::new_default(image)?);
		}

		Ok(new_self)
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
		if check_direct_buffer_write(self.memory_allocator.device().physical_device()) {
			// When possible, write directly to the new buffer memory.
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
			// If direct writes aren't possible, use a staging buffer.
			log::debug!("Allocating buffer of {data_size_bytes} bytes");
			let buf_info = BufferCreateInfo {
				usage: usage | BufferUsage::TRANSFER_DST,
				..Default::default()
			};
			let alloc_info = AllocationCreateInfo::default();
			buf = Buffer::new_slice(self.memory_allocator.clone(), buf_info, alloc_info, data_len)?;
			self.add_transfer(data, StagingDst::Buffer(buf.clone().into_bytes()), T::LAYOUT.alignment())?;
		}
		Ok(buf)
	}

	/// Create a new image from raw data.
	///
	/// `data` will be used to initialize the full extent of all mipmap levels and array layers (in
	/// that order), and it must be tightly packed together. For example, data for an image with 2
	/// mip levels and 2 array layers must be structured like:
	///
	/// - mip level 0, array layer 0
	/// - mip level 0, array layer 1
	/// - mip level 1, array layer 0
	/// - mip level 1, array layer 1
	///
	pub fn new_image<Px>(&mut self, data: &[Px], create_info: ImageCreateInfo) -> crate::Result<Arc<Image>>
	where
		Px: BufferContents + Copy,
	{
		let alloc_info = AllocationCreateInfo::default();
		let image = Image::new(self.memory_allocator.clone(), create_info, alloc_info)?;

		let alignment = image.format().block_size().try_into().unwrap();
		self.add_transfer(data, StagingDst::Image(image.clone()), alignment)?;

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
		let (format, extent, mip_levels, img_raw) = match TextureSource::File(path).load() {
			Ok(view) => view,
			Err(e) => {
				log::error!("failed to load texture: {e}");
				return Ok(self.error_texture.clone().unwrap());
			}
		};

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

	/// Load six image files into a cubemap texture. `faces` is paths to textures of each face of the
	/// cubemap, in order of +X, -X, +Y, -Y, +Z, -Z.
	///
	/// Unlike `new_texture`, the results of this are *not* cached.
	fn new_cubemap(&mut self, faces: [PathBuf; 6]) -> crate::Result<Arc<ImageView>>
	{
		let mut combined_data = Vec::<u8>::new();
		let mut cube_fmt_dim = None;
		for face_path in faces {
			let (face_fmt, face_dim, _, img_raw) = match TextureSource::File(&face_path).load() {
				Ok(ok) => ok,
				Err(e) => {
					log::error!("failed to load cubemap face texture file: {e}");
					TextureSource::Bytes(ERROR_TEXTURE_BYTES).load().unwrap()
				}
			};

			if (face_fmt, face_dim) != *cube_fmt_dim.get_or_insert((face_fmt, face_dim)) {
				return Err(EngineError::new(
					"format or extent differs between cubemap faces",
					CubemapFaceMismatch {
						existing: *cube_fmt_dim.as_ref().unwrap(),
						new_face: (face_fmt, face_dim),
					},
				));
			}

			let mip_size = get_mip_size(face_fmt, face_dim[0], face_dim[1]).try_into().unwrap();
			if combined_data.capacity() == 0 {
				combined_data.reserve(mip_size * 6);
			}
			combined_data.extend(&img_raw[..mip_size]);
		}

		let (format, extent) = cube_fmt_dim.unwrap();
		let image_info = ImageCreateInfo {
			flags: ImageCreateFlags::CUBE_COMPATIBLE,
			format,
			extent: [extent[0], extent[1], 1],
			array_layers: 6,
			usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
			..Default::default()
		};
		let image = self.new_image(&combined_data, image_info)?;

		let view_create_info = ImageViewCreateInfo {
			view_type: ImageViewType::Cube,
			..ImageViewCreateInfo::from_image(&image)
		};
		Ok(ImageView::new(image, view_create_info)?)
	}

	fn add_transfer<T>(&mut self, data: &[T], dst: StagingDst, alignment: DeviceAlignment) -> crate::Result<()>
	where
		T: BufferContents + Copy,
	{
		let data_size_bytes: DeviceSize = std::mem::size_of_val(data).try_into().unwrap();
		let nonzero_size = data_size_bytes.try_into().expect("`data` for transfer is empty");
		let transfer_layout = DeviceLayout::new(nonzero_size, alignment).unwrap();
		if transfer_layout.size() > STAGING_ARENA_SIZE {
			return Err(EngineError::new(
				"buffer/image transfer is too big",
				TransferTooBig(transfer_layout.size()),
			));
		}

		if let Some(pending_layout) = self.staging_layout {
			let (extended_usage, _) = pending_layout.extend(transfer_layout).unwrap();
			if extended_usage.size() > STAGING_ARENA_SIZE || self.transfers.len() == self.transfers.capacity() {
				log::debug!("staging arena size or transfer `Vec` capacity reached, submitting transfers now");
				self.submit_transfers()?;
			}
		}

		let (new_layout, offset) = match self.staging_layout.take() {
			Some(current_layout) => current_layout.extend(transfer_layout).unwrap(),
			None => (transfer_layout, 0),
		};

		let arena = self.staging_arenas[self.current_arena].clone();
		let staging_buf: Subbuffer<[T]> = arena.slice(offset..new_layout.size()).reinterpret();
		staging_buf.write().unwrap().copy_from_slice(data);

		self.transfers.push(StagingWork(staging_buf.into_bytes(), dst));
		self.staging_layout = Some(new_layout);

		Ok(())
	}

	/// Submit pending transfers. Run this just before building the draw command buffers so that the
	/// transfers can be done while the CPU is busy.
	fn submit_transfers(&mut self) -> crate::Result<()>
	{
		if self.staging_layout.take().is_some() {
			let queue = match self.window.transfer_queue() {
				Some(q) => q.clone(),
				None => self.window.graphics_queue().clone(),
			};
			let mut cb = AutoCommandBufferBuilder::primary(
				&self.command_buffer_allocator,
				queue.queue_family_index(),
				CommandBufferUsage::OneTimeSubmit,
			)?;

			self.transfers.drain(..).for_each(|work| work.into_cmd(&mut cb));

			self.current_arena += 1;
			if self.current_arena == self.staging_arenas.len() {
				self.current_arena = 0;
			}

			let transfer_future = if let Some(f) = self.transfer_future.take() {
				f.wait(None)?; // wait to prevent resource conflicts
				cb.build()?
					.execute_after(f, queue.clone())
					.unwrap()
					.boxed_send_sync()
					.then_signal_fence_and_flush()?
			} else {
				cb.build()?
					.execute(queue)
					.unwrap()
					.boxed_send_sync()
					.then_signal_fence_and_flush()?
			};
			self.transfer_future = Some(transfer_future);
		}

		Ok(())
	}

	/// Set the skybox using 6 texture files for each face. They'll be loaded from paths with the
	/// pattern specified in `tex_files_format` which should have an asterisk in it, for example
	/// "sky/Daylight_*.png", which will be replaced with the face name. Face names are "Right",
	/// "Left", "Top", "Bottom", "Front", and "Back".
	pub fn set_skybox(&mut self, path_pattern: String) -> crate::Result<()>
	{
		let set_layout = self.skybox_pipeline.layout().set_layouts()[0].clone();
		let face_names = ["Right", "Left", "Top", "Bottom", "Front", "Back"];
		let face_paths = face_names.map(|face_name| path_pattern.replace('*', face_name).into());
		let sky_cubemap = self.new_cubemap(face_paths)?;
		self.skybox_tex_set = Some(PersistentDescriptorSet::new(
			&self.single_set_allocator,
			set_layout,
			[WriteDescriptorSet::image_view(0, sky_cubemap)],
			[],
		)?);

		Ok(())
	}

	/// Draw the skybox to clear the color image, then get the color and depth images. They'll be
	/// resized before being cleared and returned if the window size changed.
	fn get_render_images(
		&mut self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		sky_projview: Mat4,
	) -> crate::Result<(Arc<ImageView>, Arc<ImageView>)>
	{
		let extent2 = self.window.dimensions();
		let extent = [extent2[0], extent2[1], 1];
		let (optional_extent, optional_samples) = self
			.color_image
			.as_ref()
			.map(|view| (view.image().extent(), view.image().samples()))
			.unzip();
		if Some(extent) != optional_extent || Some(self.aa_mode.sample_count()) != optional_samples {
			let rasterization_samples = self.aa_mode.sample_count();
			let aa_enabled = !matches!(self.aa_mode, AntiAliasingMode::Off);
			let color_usage = if aa_enabled {
				ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC
			} else {
				ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC
			};
			let color_create_info = ImageCreateInfo {
				format: Format::R16G16B16A16_SFLOAT,
				extent,
				samples: rasterization_samples,
				usage: color_usage,
				..Default::default()
			};
			let alloc_info = AllocationCreateInfo::default();
			let color_image = Image::new(self.memory_allocator.clone(), color_create_info, alloc_info.clone())?;
			let color = ImageView::new_default(color_image)?;
			self.color_image = Some(color.clone());

			let depth_stencil_create_info = ImageCreateInfo {
				format: self.depth_stencil_format,
				extent,
				samples: rasterization_samples,
				usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
				..Default::default()
			};
			let depth_stencil_image = Image::new(self.memory_allocator.clone(), depth_stencil_create_info, alloc_info)?;
			self.depth_stencil_image = Some(ImageView::new_default(depth_stencil_image)?);

			let set_image = if aa_enabled {
				let aa_output_create_info = ImageCreateInfo {
					format: Format::R16G16B16A16_SFLOAT,
					extent,
					usage: ImageUsage::COLOR_ATTACHMENT
						| ImageUsage::SAMPLED | ImageUsage::STORAGE
						| ImageUsage::TRANSFER_DST
						| ImageUsage::TRANSFER_SRC,
					..Default::default()
				};
				let alloc_info = AllocationCreateInfo::default();
				let aa_output = Image::new(self.memory_allocator.clone(), aa_output_create_info, alloc_info.clone())?;
				let aa_output_image = ImageView::new_default(aa_output)?;
				self.aa_output_image = Some(aa_output_image.clone());
				aa_output_image
			} else {
				self.aa_output_image = None;
				color
			};

			if matches!(self.aa_mode, AntiAliasingMode::Smaa) {
				if self.smaa_renderer.is_none() {
					self.smaa_renderer = Some(smaa::SmaaRenderer::new(self)?);
				}
			} else if self.smaa_renderer.is_some() {
				self.smaa_renderer = None;
			}

			let set_layout = self.gamma_pipeline.layout().set_layouts()[0].clone();
			let set_write = [WriteDescriptorSet::image_view(0, set_image)];
			let set = PersistentDescriptorSet::new(&self.single_set_allocator, set_layout, set_write, [])?;
			self.gamma_input_set = Some(set);
		}

		let color_image = self.color_image.clone().unwrap();
		if let Some(set) = self.skybox_tex_set.clone() {
			let viewport = Viewport {
				offset: [0.0, 0.0],
				extent: [extent[0] as f32, extent[1] as f32],
				depth_range: 0.0..=1.0,
			};
			let sky_render_info = RenderingInfo {
				color_attachments: vec![Some(RenderingAttachmentInfo {
					store_op: AttachmentStoreOp::Store,
					..RenderingAttachmentInfo::image_view(color_image.clone())
				})],
				..Default::default()
			};
			let pipeline_layout = self.skybox_pipeline.layout().clone();
			cb_builder
				.begin_rendering(sky_render_info)?
				.set_viewport(0, [viewport].as_slice().into())?
				.bind_pipeline_graphics(self.skybox_pipeline.clone())?
				.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 0, vec![set])?
				.push_constants(pipeline_layout, 0, sky_projview)?
				.draw(36, 1, 0, 0)?
				.end_rendering()?;
		}

		Ok((color_image, self.depth_stencil_image.clone().unwrap()))
	}

	fn present_image(
		&mut self,
		prior_command_buffers: SmallVec<[Arc<PrimaryAutoCommandBuffer>; 2]>,
		mut cb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
	) -> crate::Result<()>
	{
		if let Some((swapchain_image, acquire_future)) = self.window.get_next_image()? {
			let color_image = self
				.aa_output_image
				.as_ref()
				.unwrap_or_else(|| self.color_image.as_ref().unwrap())
				.image()
				.clone();

			if self.window.color_space() == ColorSpace::SrgbNonLinear {
				// perform gamma correction
				let image_extent = color_image.extent();
				let workgroups_x = image_extent[0].div_ceil(64);
				let layout = self.gamma_pipeline.layout().clone();
				cb.bind_pipeline_compute(self.gamma_pipeline.clone())?
					.bind_descriptor_sets(PipelineBindPoint::Compute, layout, 0, self.gamma_input_set.clone().unwrap())?
					.dispatch([workgroups_x, image_extent[1], 1])?;
			}

			cb.blit_image(BlitImageInfo::images(color_image, swapchain_image))?;

			let mut command_buffers = prior_command_buffers;
			command_buffers.push(cb.build()?);

			// wait for any transfers to finish
			// (ideally we'd use a semaphore, but it's borked in Vulkano right now)
			if let Some(f) = &self.transfer_future {
				f.wait(None)?;
			}
			let transfer_future = self.transfer_future.take();

			// submit the built command buffers, presenting it if possible
			self.window.present(command_buffers, acquire_future, transfer_future)?;
		}

		Ok(())
	}

	pub fn graphics_queue_family_index(&self) -> u32
	{
		self.window.graphics_queue().queue_family_index()
	}

	/// Check if the window has been resized since the last frame submission.
	pub fn window_resized(&self) -> bool
	{
		self.window.extent_changed()
	}

	pub fn set_fullscreen(&self, fullscreen: bool)
	{
		self.window.set_fullscreen(fullscreen)
	}
	pub fn is_fullscreen(&self) -> bool
	{
		self.window.is_fullscreen()
	}

	pub fn handle_window_event(&mut self, window_event: &mut WindowEvent)
	{
		self.window.handle_window_event(window_event)
	}
	pub fn window_dimensions(&self) -> [u32; 2]
	{
		self.window.dimensions()
	}

	/// Get the delta time for last frame.
	pub fn delta(&self) -> std::time::Duration
	{
		self.window.delta()
	}
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum AntiAliasingMode
{
	#[default]
	Off,
	Multisample2,
	Multisample4,
	Smaa,
}
impl AntiAliasingMode
{
	fn sample_count(&self) -> SampleCount
	{
		match self {
			Self::Multisample2 => SampleCount::Sample2,
			Self::Multisample4 => SampleCount::Sample4,
			_ => SampleCount::Sample1,
		}
	}
}

#[derive(Debug)]
struct CubemapFaceMismatch
{
	existing: (Format, [u32; 2]),
	new_face: (Format, [u32; 2]),
}
impl std::error::Error for CubemapFaceMismatch {}
impl std::fmt::Display for CubemapFaceMismatch
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		write!(
			f,
			"existing cubemap face has format and extent of {:?}, but attempted to add face with {:?}",
			self.existing, self.new_face,
		)
	}
}

#[derive(Debug)]
struct TransferTooBig(DeviceSize);
impl std::error::Error for TransferTooBig {}
impl std::fmt::Display for TransferTooBig
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		write!(
			f,
			"buffer/image transfers must be no larger than {} bytes, but the transfer is {} bytes",
			STAGING_ARENA_SIZE, self.0,
		)
	}
}

/// Check if we can directly write to device-local buffer memory, as doing so may be faster.
fn check_direct_buffer_write(physical_device: &Arc<PhysicalDevice>) -> bool
{
	match physical_device.properties().device_type {
		PhysicalDeviceType::IntegratedGpu => true, // Always possible for integrated GPU.
		_ => {
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
	}
}

fn create_sky_pipeline(
	pipeline_layout: Arc<PipelineLayout>,
	rasterization_samples: SampleCount,
) -> crate::Result<Arc<GraphicsPipeline>>
{
	let device = pipeline_layout.device().clone();
	let rendering_formats = PipelineRenderingCreateInfo {
		color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
		..Default::default()
	};
	let pipeline_info = GraphicsPipelineCreateInfo {
		stages: smallvec::smallvec![
			PipelineShaderStageCreateInfo::new(sky_vs::load(device.clone())?.entry_point("main").unwrap()),
			PipelineShaderStageCreateInfo::new(sky_fs::load(device.clone())?.entry_point("main").unwrap()),
		],
		vertex_input_state: Some(Default::default()),
		input_assembly_state: Some(Default::default()),
		viewport_state: Some(Default::default()),
		rasterization_state: Some(Default::default()),
		multisample_state: Some(MultisampleState {
			rasterization_samples,
			..Default::default()
		}),
		color_blend_state: Some(ColorBlendState::with_attachment_states(1, Default::default())),
		dynamic_state: [DynamicState::Viewport].into_iter().collect(),
		subpass: Some(rendering_formats.into()),
		..GraphicsPipelineCreateInfo::layout(pipeline_layout)
	};
	Ok(GraphicsPipeline::new(device, None, pipeline_info)?)
}
mod sky_vs
{
	vulkano_shaders::shader! {
		ty: "vertex",
		path: "src/shaders/sky.vert.glsl",
	}
}
mod sky_fs
{
	vulkano_shaders::shader! {
		ty: "fragment",
		path: "src/shaders/sky.frag.glsl",
	}
}

fn create_gamma_pipeline(device: Arc<Device>) -> crate::Result<Arc<ComputePipeline>>
{
	let color_image_binding = DescriptorSetLayoutBinding {
		stages: ShaderStages::COMPUTE,
		..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
	};
	let color_set_layout_info = DescriptorSetLayoutCreateInfo {
		bindings: [(0, color_image_binding)].into(),
		..Default::default()
	};
	let color_set_layout = DescriptorSetLayout::new(device.clone(), color_set_layout_info)?;

	let gamma_pipeline_layout_info = PipelineLayoutCreateInfo {
		set_layouts: vec![color_set_layout],
		..Default::default()
	};
	let gamma_pipeline_layout = PipelineLayout::new(device.clone(), gamma_pipeline_layout_info)?;
	let gamma_entry_point = compute_gamma::load(device.clone())?.entry_point("main").unwrap();
	let gamma_stage = PipelineShaderStageCreateInfo::new(gamma_entry_point);
	let gamma_pipeline_create_info = ComputePipelineCreateInfo::stage_layout(gamma_stage, gamma_pipeline_layout);
	Ok(ComputePipeline::new(device, None, gamma_pipeline_create_info)?)
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

enum StagingDst
{
	Buffer(Subbuffer<[u8]>),
	Image(Arc<Image>),
}
struct StagingWork(Subbuffer<[u8]>, StagingDst);
impl StagingWork
{
	fn into_cmd(self, cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
	{
		let Self(src, dst) = self;
		match dst {
			StagingDst::Buffer(dst_buf) => cb.copy_buffer(CopyBufferInfo::buffers(src, dst_buf)),
			StagingDst::Image(dst_image) => {
				let format = dst_image.format();
				let mip_levels = dst_image.mip_levels();
				let array_layers = dst_image.array_layers();

				// generate copies for every mipmap level
				let mut regions = SmallVec::with_capacity(mip_levels as usize);
				let [mut mip_w, mut mip_h, _] = dst_image.extent();
				let mut buffer_offset: DeviceSize = 0;
				for mip_level in 0..mip_levels {
					regions.push(BufferImageCopy {
						buffer_offset,
						image_subresource: ImageSubresourceLayers {
							mip_level,
							..ImageSubresourceLayers::from_parameters(format, array_layers)
						},
						image_extent: [mip_w, mip_h, 1],
						..Default::default()
					});

					buffer_offset += get_mip_size(format, mip_w, mip_h) * (array_layers as DeviceSize);
					mip_w /= 2;
					mip_h /= 2;
				}

				let copy_info = CopyBufferToImageInfo {
					regions,
					..CopyBufferToImageInfo::buffer_image(src, dst_image)
				};
				cb.copy_buffer_to_image(copy_info)
			}
		}
		.unwrap();
	}
}

//
/* Texture stuff */
//
#[derive(Copy, Clone)]
pub enum TextureSource<'a>
{
	File(&'a Path),  // Open and read the file at the given path.
	Bytes(&'a [u8]), // Read the file from a byte slice.
}
impl TextureSource<'_>
{
	fn load(self) -> Result<(Format, [u32; 2], u32, Vec<u8>), TextureLoadingError>
	{
		self.load_inner().map_err(|error| {
			let file_path = match self {
				Self::File(path) => path.to_string_lossy().into_owned().into(),
				Self::Bytes(_) => "<texture file byte slice>".into(),
			};
			TextureLoadingError { file_path, error }
		})
	}

	fn load_inner(self) -> Result<(Format, [u32; 2], u32, Vec<u8>), TexLoadErrVariant>
	{
		let seekread: Box<dyn SeekRead> = match self {
			Self::File(path) => Box::new(std::fs::File::open(path)?),
			Self::Bytes(data) => Box::new(Cursor::new(data)),
		};
		let mut reader = BufReader::new(seekread);

		let mut magic_number = [0; 4];
		reader.read_exact(&mut magic_number)?;
		reader.rewind()?;

		match std::str::from_utf8(&magic_number) {
			Ok("DDS ") => {
				let dds = ddsfile::Dds::read(reader)?;

				// BC7_UNorm is treated as sRGB for now since Compressonator can't output
				// BC7_UNorm_sRGB, even though the data itself appears to be in sRGB gamma.
				let format = match dds.get_dxgi_format() {
					Some(DxgiFormat::BC1_UNorm_sRGB) => Format::BC1_RGBA_SRGB_BLOCK,
					Some(DxgiFormat::BC4_UNorm) => Format::BC4_UNORM_BLOCK,
					Some(DxgiFormat::BC4_SNorm) => Format::BC4_SNORM_BLOCK,
					Some(DxgiFormat::BC5_UNorm) => Format::BC5_UNORM_BLOCK,
					Some(DxgiFormat::BC5_SNorm) => Format::BC5_SNORM_BLOCK,
					Some(DxgiFormat::BC7_UNorm) => Format::BC7_SRGB_BLOCK,
					Some(DxgiFormat::BC7_UNorm_sRGB) => Format::BC7_SRGB_BLOCK,
					format => return Err(TexLoadErrVariant::DdsFormat(format)),
				};
				let extent = [dds.get_width(), dds.get_height()];
				let mip_levels = dds.get_num_mipmap_levels();

				Ok((format, extent, mip_levels, dds.data))
			}
			_ => {
				let img = image::io::Reader::new(reader).with_guessed_format()?.decode()?.into_rgba8();
				Ok((Format::R8G8B8A8_SRGB, img.dimensions().into(), 1, img.into_raw()))
			}
		}
	}
}
trait SeekRead: Seek + Read {}
impl<T: Seek + Read> SeekRead for T {}

#[derive(Debug)]
enum TexLoadErrVariant
{
	DdsFormat(Option<DxgiFormat>),
	Other(Box<dyn std::error::Error>),
}
impl<E: std::error::Error + 'static> From<E> for TexLoadErrVariant
{
	fn from(e: E) -> Self
	{
		Self::Other(Box::new(e))
	}
}
impl std::fmt::Display for TexLoadErrVariant
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		match self {
			Self::DdsFormat(Some(format)) => write!(f, "format '{format:?}' is unsupported"),
			Self::DdsFormat(None) => write!(f, "DDS file doesn't have a DXGI format"),
			Self::Other(e) => write!(f, "{e}"),
		}
	}
}

#[derive(Debug)]
struct TextureLoadingError
{
	file_path: Cow<'static, str>,
	error: TexLoadErrVariant,
}
impl std::error::Error for TextureLoadingError {}
impl std::fmt::Display for TextureLoadingError
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "{}: {}", self.file_path, self.error)
	}
}

/// Calculate the size (in bytes) that a mip level with the given format and extent would take up.
///
/// This does not take array layers into account; the returned value must be multiplied by the
/// array layer count to get the total size across all layers.
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
pub(crate) fn prepare_rendering(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut mesh_manager: UniqueViewMut<MeshManager>,
) -> crate::Result<()>
{
	// Take this opportunity to re-create pipelines (that render to multisample images) if
	// rasterization samples changed, since this system is run before command buffers are built.
	if let Some(prev_color_image) = render_ctx.color_image.as_ref() {
		let new_sample_count = render_ctx.aa_mode.sample_count();
		if new_sample_count != prev_color_image.image().samples() {
			let sky_pipeline_layout = render_ctx.skybox_pipeline.layout().clone();
			render_ctx.skybox_pipeline = create_sky_pipeline(sky_pipeline_layout, new_sample_count)?;
			mesh_manager.change_rasterization_samples(new_sample_count)?;
		}
	}

	render_ctx.submit_transfers()
}

// Submit all the command buffers for this frame to present the results.
pub(crate) fn submit_frame(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut mesh_manager: UniqueViewMut<MeshManager>,
	mut canvas: UniqueViewMut<Canvas>,
	mut light_manager: UniqueViewMut<LightManager>,
	camera_manager: UniqueView<CameraManager>,
) -> crate::Result<()>
{
	// A minimized window sometimes reports an inner width or height of 0, which we can't resize
	// the swapchain to. Presenting anyways causes an "out of date" error, so just don't present.
	if render_ctx.window.is_minimized() {
		return Ok(());
	}

	let mut cb_builder = AutoCommandBufferBuilder::primary(
		&render_ctx.command_buffer_allocator,
		render_ctx.graphics_queue_family_index(),
		CommandBufferUsage::OneTimeSubmit,
	)?;

	light_manager.update_buffer(&mut cb_builder)?;

	let (color_image, depth_stencil_image) = render_ctx.get_render_images(&mut cb_builder, camera_manager.sky_projview())?;

	// opaque 3D objects
	mesh_manager.execute_rendering(&mut cb_builder, color_image.clone(), depth_stencil_image.clone())?;

	// transparent 3D objects (OIT)
	let memory_allocator = render_ctx.memory_allocator.clone();
	render_ctx.transparency_renderer.process_transparency(
		&mut cb_builder,
		color_image.clone(),
		depth_stencil_image.clone(),
		memory_allocator.clone(),
	)?;

	let aa_output = match render_ctx.aa_mode {
		AntiAliasingMode::Off => color_image,
		AntiAliasingMode::Smaa => {
			let aa_output_image = render_ctx.aa_output_image.clone().unwrap();
			let smaa = render_ctx.smaa_renderer.as_mut().unwrap();
			smaa.run(
				&mut cb_builder,
				memory_allocator,
				color_image,
				depth_stencil_image,
				aa_output_image.clone(),
			)?;
			aa_output_image
		}
		_ => {
			let aa_output_image = render_ctx.aa_output_image.clone().unwrap();
			cb_builder.resolve_image(ResolveImageInfo::images(
				color_image.image().clone(),
				aa_output_image.image().clone(),
			))?;
			aa_output_image
		}
	};

	// UI
	canvas.execute_rendering(&mut cb_builder, aa_output)?;

	let mut prior_command_buffers = SmallVec::new();
	if let Some(light_cb) = light_manager.take_light_cb() {
		prior_command_buffers.push(light_cb);
	}
	render_ctx.present_image(prior_command_buffers, cb_builder)
}
