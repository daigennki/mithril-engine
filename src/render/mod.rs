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
pub mod workload;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddsfile::DxgiFormat;
use glam::*;

use vulkano::buffer::{subbuffer::Subbuffer, Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{
	allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
};
use vulkano::device::DeviceOwned;
use vulkano::format::{Format, NumericType};
use vulkano::image::{
	sampler::{Filter, Sampler, SamplerCreateInfo},
	view::ImageView,
	Image, ImageCreateInfo, ImageUsage,
};
use vulkano::memory::{
	allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
	MemoryPropertyFlags,
};
use vulkano::pipeline::graphics::{
	color_blend::ColorBlendState,
	depth_stencil::{CompareOp, DepthStencilState},
	input_assembly::InputAssemblyState,
	multisample::MultisampleState,
	rasterization::RasterizationState,
	subpass::PipelineRenderingCreateInfo,
	vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState},
	viewport::ViewportState,
	GraphicsPipelineCreateInfo,
};
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::{spirv::ExecutionModel, EntryPoint, ShaderInterfaceEntryType, ShaderStages};
use vulkano::DeviceSize;

use crate::EngineError;

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	swapchain: swapchain::Swapchain,
	main_render_target: render_target::RenderTarget,
	memory_allocator: Arc<StandardMemoryAllocator>,
	descriptor_set_allocator: StandardDescriptorSetAllocator,
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

		let set_alloc_info = StandardDescriptorSetAllocatorCreateInfo::default();
		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(vk_dev.clone(), set_alloc_info);
		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_dev.clone()));

		// The counts below are multiplied by the number of swapchain images, to account for previous submissions.
		// - Primary: One for graphics, another for async transfers, each on separate queue families.
		// - Secondary: Only up to four should be created per thread.
		let cb_alloc_info = StandardCommandBufferAllocatorCreateInfo {
			primary_buffer_count: swapchain.image_count(),
			secondary_buffer_count: 4 * swapchain.image_count(),
			..Default::default()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(vk_dev.clone(), cb_alloc_info);

		let main_render_target = render_target::RenderTarget::new(
			memory_allocator.clone(),
			&descriptor_set_allocator,
			swapchain.get_images(),
			swapchain.color_space(),
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

		let transfer_manager = transfer::TransferManager::new(transfer_queue, memory_allocator.clone());

		Ok(RenderContext {
			swapchain,
			memory_allocator,
			descriptor_set_allocator,
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
			&self.descriptor_set_allocator,
			material_textures_set_layout,
			self.swapchain.dimensions(),
			self.depth_stencil_format(),
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

	fn submit_async_transfers(&mut self) -> crate::Result<()>
	{
		self.transfer_manager.submit_async_transfers(&self.command_buffer_allocator)
	}

	fn resize_everything_else(&mut self) -> crate::Result<()>
	{
		// Update images to match the current swapchain image extent.
		self.main_render_target.resize(
			self.memory_allocator.clone(),
			&self.descriptor_set_allocator,
			self.swapchain.get_images(),
			self.swapchain.color_space(),
		)?;
		if let Some(transparency_renderer) = &mut self.transparency_renderer {
			transparency_renderer.resize_image(
				self.memory_allocator.clone(),
				&self.descriptor_set_allocator,
				self.swapchain.dimensions(),
			)?
		}

		Ok(())
	}

	fn depth_stencil_format(&self) -> Format
	{
		self.main_render_target.depth_image().format()
	}

	fn submit_primary(&mut self, built_cb: Arc<PrimaryAutoCommandBuffer>) -> crate::Result<()>
	{
		self.swapchain.submit(built_cb, self.transfer_manager.take_transfer_future())
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

fn get_mip_size(format: Format, mip_width: u32, mip_height: u32) -> DeviceSize
{
	let block_extent = format.block_extent();
	let block_size = format.block_size();
	let x_blocks = mip_width.div_ceil(block_extent[0]) as DeviceSize;
	let y_blocks = mip_height.div_ceil(block_extent[1]) as DeviceSize;
	x_blocks * y_blocks * block_size
}

/// Create a new graphics pipeline using the given parameters.
pub fn new_graphics_pipeline(
	input_assembly_state: InputAssemblyState,
	stage_entry_points: &[EntryPoint],
	rasterization_state: RasterizationState,
	pipeline_layout: Arc<PipelineLayout>,
	rendering_info: PipelineRenderingCreateInfo,
	color_blend_state: Option<ColorBlendState>,
	depth_stencil_state: Option<DepthStencilState>,
) -> crate::Result<Arc<GraphicsPipeline>>
{
	let vertex_input_state = stage_entry_points
		.iter()
		.find(|entry_point| entry_point.info().execution_model == ExecutionModel::Vertex)
		.map(gen_vertex_input_state);

	let stages = stage_entry_points
		.iter()
		.cloned()
		.map(PipelineShaderStageCreateInfo::new)
		.collect();

	let device = pipeline_layout.device().clone();
	let pipeline_info = GraphicsPipelineCreateInfo {
		stages,
		vertex_input_state,
		input_assembly_state: Some(input_assembly_state),
		viewport_state: Some(ViewportState::default()),
		rasterization_state: Some(rasterization_state),
		multisample_state: Some(MultisampleState::default()),
		depth_stencil_state,
		color_blend_state,
		dynamic_state: [DynamicState::Viewport].into_iter().collect(),
		subpass: Some(rendering_info.into()),
		..GraphicsPipelineCreateInfo::layout(pipeline_layout)
	};
	Ok(GraphicsPipeline::new(device, None, pipeline_info)?)
}

/// Automatically determine vertex inputs using information from the given vertex shader module.
fn gen_vertex_input_state(entry_point: &EntryPoint) -> VertexInputState
{
	let (bindings, attributes) = entry_point
		.info()
		.input_interface
		.elements()
		.iter()
		.map(|input| {
			let binding = input.location;
			let attribute_desc = VertexInputAttributeDescription {
				binding,
				format: format_from_shader_interface(&input.ty),
				offset: 0,
			};
			let binding_desc = VertexInputBindingDescription {
				stride: attribute_desc.format.block_size().try_into().unwrap(),
				input_rate: VertexInputRate::Vertex,
			};
			((binding, binding_desc), (binding, attribute_desc))
		})
		.unzip();

	VertexInputState {
		bindings,
		attributes,
		..Default::default()
	}
}
fn format_from_shader_interface(ty: &ShaderInterfaceEntryType) -> Format
{
	let possible_formats = match ty.base_type {
		NumericType::Float => [
			Format::R32_SFLOAT,
			Format::R32G32_SFLOAT,
			Format::R32G32B32_SFLOAT,
			Format::R32G32B32A32_SFLOAT,
		],
		NumericType::Int => [
			Format::R32_SINT,
			Format::R32G32_SINT,
			Format::R32G32B32_SINT,
			Format::R32G32B32A32_SINT,
		],
		NumericType::Uint => [
			Format::R32_UINT,
			Format::R32G32_UINT,
			Format::R32G32B32_UINT,
			Format::R32G32B32A32_UINT,
		],
	};
	let format_index = (ty.num_components - 1) as usize;
	possible_formats[format_index]
}
