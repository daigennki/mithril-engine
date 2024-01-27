/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod lighting;
pub mod model;
mod render_target;
pub mod skybox;
mod swapchain;
mod transfer;
mod transparency;
pub mod ui;
mod vulkan_init;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddsfile::DxgiFormat;
use glam::*;
use shipyard::{IntoWorkload, UniqueView, UniqueViewMut, Workload};

use vulkano::buffer::{subbuffer::Subbuffer, Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	AutoCommandBufferBuilder, CommandBufferUsage,
};
use vulkano::descriptor_set::layout::{
	DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{
	sampler::{Filter, Sampler, SamplerCreateInfo},
	view::ImageView,
	Image, ImageCreateInfo, ImageUsage,
};
use vulkano::memory::{
	allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
	MemoryPropertyFlags,
};
use vulkano::pipeline::graphics::depth_stencil::CompareOp;
use vulkano::shader::ShaderStages;
use vulkano::DeviceSize;

use crate::component::camera::CameraManager;
use crate::EngineError;
use lighting::LightManager;
use model::MeshManager;
use ui::Canvas;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	swapchain: swapchain::Swapchain,
	main_render_target: render_target::RenderTarget,
	memory_allocator: Arc<StandardMemoryAllocator>,
	command_buffer_allocator: StandardCommandBufferAllocator,

	transparency_renderer: Option<transparency::MomentTransparencyRenderer>,

	light_set_layout: Arc<DescriptorSetLayout>,

	// Loaded textures, with the key being the path relative to the current working directory
	textures: HashMap<PathBuf, Arc<ImageView>>,

	allow_direct_buffer_access: bool,

	transfer_manager: transfer::TransferManager,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) -> crate::Result<Self>
	{
		let (graphics_queue, transfer_queue, allow_direct_buffer_access) = vulkan_init::vulkan_setup(game_name, event_loop)?;
		let vk_dev = graphics_queue.device().clone();

		let swapchain = swapchain::Swapchain::new(graphics_queue, event_loop, game_name)?;

		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_dev.clone()));

		// - Primary command buffers: One for each graphics submission.
		// - Secondary command buffers: Only up to four should be created per thread.
		let cb_alloc_info = StandardCommandBufferAllocatorCreateInfo {
			primary_buffer_count: swapchain.image_count(),
			secondary_buffer_count: 4,
			..Default::default()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(vk_dev.clone(), cb_alloc_info);

		let main_render_target = render_target::RenderTarget::new(memory_allocator.clone(), swapchain.dimensions())?;

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

		let transfer_manager = transfer::TransferManager::new(transfer_queue, memory_allocator.clone());

		Ok(RenderContext {
			swapchain,
			memory_allocator,
			command_buffer_allocator,
			main_render_target,
			transparency_renderer: None,
			light_set_layout,
			textures: HashMap::new(),
			allow_direct_buffer_access,
			transfer_manager,
		})
	}

	fn load_transparency(&mut self, material_textures_set_layout: Arc<DescriptorSetLayout>) -> crate::Result<()>
	{
		self.transparency_renderer = Some(transparency::MomentTransparencyRenderer::new(
			self.memory_allocator.clone(),
			material_textures_set_layout,
			self.swapchain.dimensions(),
			self.main_render_target.depth_stencil_format(),
		)?);
		Ok(())
	}

	/// Create a device-local buffer from a slice, initialized with `data` for `usage`.
	/// For stuff that isn't an array, just put the data into a single-element slice, like `[data]`.
	fn new_buffer<T>(&mut self, data: Vec<T>, usage: BufferUsage) -> crate::Result<Subbuffer<[T]>>
	where
		T: BufferContents + Copy,
	{
		let data_len = data.len().try_into().unwrap();
		let data_size_bytes = data.len() * std::mem::size_of::<T>();
		let buf;
		if self.allow_direct_buffer_access {
			// When possible, upload directly to the new buffer memory.
			log::debug!("Allocating direct buffer of {} bytes", data_size_bytes);
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
			buf.write().unwrap().copy_from_slice(&data);
		} else {
			// If direct uploads aren't possible, create a staging buffer on the CPU side,
			// then submit a transfer command to the new buffer on the GPU side.
			log::debug!("Allocating buffer of {} bytes", data_size_bytes);
			let buf_info = BufferCreateInfo {
				usage: usage | BufferUsage::TRANSFER_DST,
				..Default::default()
			};
			let alloc_info = AllocationCreateInfo::default();
			buf = Buffer::new_slice(self.memory_allocator.clone(), buf_info, alloc_info, data_len)?;
			self.transfer_manager.copy_to_buffer(data, buf.clone());
		}
		Ok(buf)
	}

	pub fn new_image<Px>(&mut self, data: Vec<Px>, create_info: ImageCreateInfo) -> crate::Result<Arc<Image>>
	where
		Px: BufferContents + Copy,
	{
		let alloc_info = AllocationCreateInfo::default();
		let image = Image::new(self.memory_allocator.clone(), create_info, alloc_info)?;

		self.transfer_manager.copy_to_image(data, image.clone());

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
		let image = self.new_image(img_raw, image_info)?;
		let view = ImageView::new_default(image)?;

		self.textures.insert(path.to_path_buf(), view.clone());

		Ok(view)
	}

	fn graphics_queue_family_index(&self) -> u32
	{
		self.swapchain.graphics_queue_family_index()
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

	/// Get the delta time for last frame.
	pub fn delta(&self) -> std::time::Duration
	{
		self.swapchain.delta()
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
	(submit_async_transfers, model::draw_workload, draw_ui, submit_frame).into_workload()
}

fn submit_async_transfers(mut render_ctx: UniqueViewMut<RenderContext>) -> crate::Result<()>
{
	render_ctx.transfer_manager.submit_async_transfers()
}
fn draw_ui(render_ctx: UniqueView<RenderContext>, mut canvas: UniqueViewMut<Canvas>) -> crate::Result<()>
{
	canvas.draw(&render_ctx)
}

// Submit all the command buffers for this frame to actually render them to the image.
fn submit_frame(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut skybox: UniqueViewMut<skybox::Skybox>,
	mut mesh_manager: UniqueViewMut<MeshManager>,
	mut canvas: UniqueViewMut<Canvas>,
	light_manager: UniqueView<LightManager>,
	camera_manager: UniqueView<CameraManager>,
) -> crate::Result<()>
{
	let mut primary_cb_builder = AutoCommandBufferBuilder::primary(
		&render_ctx.command_buffer_allocator,
		render_ctx.graphics_queue_family_index(),
		CommandBufferUsage::OneTimeSubmit,
	)?;

	render_ctx
		.transfer_manager
		.add_synchronous_transfer_commands(&mut primary_cb_builder)?;

	// Sometimes no image may be returned because the image is out of date or the window is
	// minimized, in which case, don't present.
	if let Some(swapchain_image) = render_ctx.swapchain.get_next_image()? {
		let memory_allocator = render_ctx.memory_allocator.clone();
		let swapchain_extent = render_ctx.swapchain.dimensions();
		let (color_image, depth_image) = render_ctx
			.main_render_target
			.get_images(memory_allocator.clone(), swapchain_extent)?;

		// shadows
		light_manager.execute_shadow_rendering(&mut primary_cb_builder)?;

		// skybox (effectively clears the image)
		skybox.draw(
			&mut primary_cb_builder,
			color_image.clone(),
			camera_manager.sky_projview().as_mat4(),
		)?;

		// 3D
		mesh_manager.execute_rendering(&mut primary_cb_builder, color_image.clone(), depth_image.clone())?;

		// 3D OIT
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

		// blit the image to the swapchain image, converting it to the swapchain's color space if necessary
		render_ctx.main_render_target.blit_to_swapchain(
			&mut primary_cb_builder,
			swapchain_image,
			render_ctx.swapchain.color_space(),
		)?;
	}

	// submit the built command buffer, presenting it if possible
	let built_cb = primary_cb_builder.build()?;
	let transfer_future = render_ctx.transfer_manager.take_transfer_future();
	render_ctx.swapchain.submit(built_cb, transfer_future)
}
