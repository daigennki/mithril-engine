/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod lighting;
pub mod model;
mod transfer;
mod transparency;
pub mod ui;
mod window;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddsfile::DxgiFormat;
use glam::*;
use shipyard::{IntoWorkload, UniqueView, UniqueViewMut, Workload};

use vulkano::buffer::{subbuffer::Subbuffer, Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderingAttachmentInfo,
	RenderingInfo,
};
use vulkano::descriptor_set::{
	allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{
	physical::{PhysicalDevice, PhysicalDeviceType},
	Device, DeviceOwned,
};
use vulkano::format::{Format, FormatFeatures};
use vulkano::image::{
	sampler::{Sampler, SamplerCreateInfo},
	view::{ImageView, ImageViewCreateInfo, ImageViewType},
	Image, ImageCreateFlags, ImageCreateInfo, ImageUsage,
};
use vulkano::memory::{
	allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
	MemoryPropertyFlags,
};
use vulkano::pipeline::{
	compute::ComputePipelineCreateInfo,
	graphics::{
		color_blend::ColorBlendState,
		input_assembly::{InputAssemblyState, PrimitiveTopology},
		subpass::PipelineRenderingCreateInfo,
		viewport::Viewport,
		GraphicsPipelineCreateInfo,
	},
	layout::{PipelineLayoutCreateInfo, PushConstantRange},
	ComputePipeline, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
	PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::AttachmentStoreOp;
use vulkano::shader::ShaderStages;
use vulkano::swapchain::ColorSpace;
use vulkano::DeviceSize;
use winit::event_loop::EventLoop;

use crate::component::camera::CameraManager;
use crate::EngineError;
use lighting::LightManager;
use model::MeshManager;
use ui::Canvas;

// Some notes regarding observed support (with AMD, Intel, and NVIDIA GPUs) for depth/stencil formats:
//
// - `D16_UNORM`: Supported on all GPUs.
// - `D16_UNORM_S8_UINT`: Only supported on AMD GPUs.
// - `X8_D24_UNORM_PACK32`: Only supported on NVIDIA and Intel GPUs.
// - `D24_UNORM_S8_UINT`: Only supported on NVIDIA and Intel GPUs.
// - `D32_SFLOAT`: Supported on all GPUs.
// - `D32_SFLOAT_S8_UINT`: Supported on all GPUs.
// - `S8_UINT`: Only supported on AMD GPUs. Possibly supported on NVIDIA GPUs.
//
// (source: https://vulkan.gpuinfo.org/listoptimaltilingformats.php)
const DEPTH_STENCIL_FORMAT_CANDIDATES: [Format; 2] = [Format::D24_UNORM_S8_UINT, Format::D16_UNORM_S8_UINT];

#[derive(shipyard::Unique)]
pub struct RenderContext
{
	window: window::GameWindow,
	memory_allocator: Arc<StandardMemoryAllocator>,
	command_buffer_allocator: StandardCommandBufferAllocator,
	single_set_allocator: StandardDescriptorSetAllocator,
	direct_buffer_write: bool,
	transfer_manager: transfer::TransferManager,
	skybox_pipeline: Arc<GraphicsPipeline>,
	skybox_tex_set: Option<Arc<PersistentDescriptorSet>>,
	transparency_renderer: Option<transparency::MomentTransparencyRenderer>,

	// Things related to the output color/depth images and gamma correction.
	depth_stencil_format: Format,
	depth_image: Option<Arc<ImageView>>,
	color_image: Option<Arc<ImageView>>,
	color_set: Option<Arc<PersistentDescriptorSet>>, // Contains `color_image` as a storage image binding.
	gamma_pipeline: Arc<ComputePipeline>,            // The gamma correction pipeline.

	// Loaded textures, with the key being the path relative to the current working directory.
	textures: HashMap<PathBuf, Arc<ImageView>>,
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &EventLoop<()>) -> crate::Result<Self>
	{
		let window = window::GameWindow::new(event_loop, game_name)?;
		let vk_dev = window.graphics_queue().device().clone();
		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(vk_dev.clone()));

		// Check if we can directly write to device-local buffer memory, as doing so may be faster.
		let direct_buffer_write = check_direct_buffer_write(vk_dev.physical_device());
		if direct_buffer_write {
			log::info!("Enabling direct buffer writes.");
		}

		// - Primary command buffers: One for each graphics submission.
		// - Secondary command buffers: Only up to four should be created per thread.
		let cb_alloc_info = StandardCommandBufferAllocatorCreateInfo {
			primary_buffer_count: window.image_count(),
			secondary_buffer_count: 4,
			..Default::default()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(vk_dev.clone(), cb_alloc_info);

		let transfer_queue = window.transfer_queue().clone();
		let submit_transfers_to = transfer_queue.unwrap_or_else(|| window.graphics_queue().clone());
		let transfer_manager = transfer::TransferManager::new(submit_transfers_to, memory_allocator.clone())?;

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
			.ok_or("none of the depth/stencil format candidates are supported")?;
		log::debug!("using depth/stencil format {depth_stencil_format:?}");

		let color_image_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::COMPUTE,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
		};
		let color_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [(0, color_image_binding)].into(),
			..Default::default()
		};
		let color_set_layout = DescriptorSetLayout::new(vk_dev.clone(), color_set_layout_info)?;

		let gamma_pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![color_set_layout],
			..Default::default()
		};
		let gamma_pipeline_layout = PipelineLayout::new(vk_dev.clone(), gamma_pipeline_layout_info)?;
		let gamma_entry_point = compute_gamma::load(vk_dev.clone())?.entry_point("main").unwrap();
		let gamma_stage = PipelineShaderStageCreateInfo::new(gamma_entry_point);
		let gamma_pipeline_create_info = ComputePipelineCreateInfo::stage_layout(gamma_stage, gamma_pipeline_layout);
		let gamma_pipeline = ComputePipeline::new(vk_dev.clone(), None, gamma_pipeline_create_info)?;

		let skybox_pipeline = create_sky_pipeline(vk_dev.clone())?;

		Ok(Self {
			window,
			memory_allocator,
			command_buffer_allocator,
			single_set_allocator,
			direct_buffer_write,
			transfer_manager,
			skybox_pipeline,
			skybox_tex_set: None,
			transparency_renderer: None,
			depth_stencil_format,
			depth_image: None,
			color_image: None,
			color_set: None,
			gamma_pipeline,
			textures: HashMap::new(),
		})
	}

	fn load_transparency(&mut self, material_textures_set_layout: Arc<DescriptorSetLayout>) -> crate::Result<()>
	{
		self.transparency_renderer = Some(transparency::MomentTransparencyRenderer::new(
			self.memory_allocator.clone(),
			material_textures_set_layout,
			self.window.dimensions(),
			self.depth_stencil_format,
		)?);
		Ok(())
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
		if self.direct_buffer_write {
			// When possible, upload directly to the new buffer memory.
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
			// If direct uploads aren't possible, create a staging buffer on the CPU side,
			// then submit a transfer command to the new buffer on the GPU side.
			log::debug!("Allocating buffer of {data_size_bytes} bytes");
			let buf_info = BufferCreateInfo {
				usage: usage | BufferUsage::TRANSFER_DST,
				..Default::default()
			};
			let alloc_info = AllocationCreateInfo::default();
			buf = Buffer::new_slice(self.memory_allocator.clone(), buf_info, alloc_info, data_len)?;
			self.transfer_manager.copy_to_buffer(data, buf.clone())?;
		}
		Ok(buf)
	}

	pub fn new_image<Px>(&mut self, data: &[Px], create_info: ImageCreateInfo) -> crate::Result<Arc<Image>>
	where
		Px: BufferContents + Copy,
	{
		let alloc_info = AllocationCreateInfo::default();
		let image = Image::new(self.memory_allocator.clone(), create_info, alloc_info)?;

		self.transfer_manager.copy_to_image(data, image.clone())?;

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
		let mut cube_fmt = None;
		let mut cube_dim = None;
		for face_path in faces {
			let (face_fmt, face_dim, _, img_raw) = load_texture(&face_path)?;

			if face_fmt != *cube_fmt.get_or_insert(face_fmt) {
				return Err("Not all faces of a cubemap have the same format!".into());
			}
			if face_dim != *cube_dim.get_or_insert(face_dim) {
				return Err("Not all faces of a cubemap have the same dimensions!".into());
			}

			let mip_size = get_mip_size(face_fmt, face_dim[0], face_dim[1]).try_into().unwrap();
			if combined_data.capacity() == 0 {
				combined_data.reserve(mip_size * 6);
			}
			combined_data.extend(&img_raw[..mip_size]);
		}

		let extent = cube_dim.unwrap();
		let image_info = ImageCreateInfo {
			flags: ImageCreateFlags::CUBE_COMPATIBLE,
			format: cube_fmt.unwrap(),
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

	/// Set the skybox using 6 texture files for each face. They'll be loaded from paths with the
	/// pattern specified in `tex_files_format` which should have an asterisk in it, for example
	/// "sky/Daylight Box_*.png", which will be replaced with the face name. Face names are "Right",
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

	/// Get the color and depth images. They'll be resized before being returned if the window size
	/// changed.
	fn get_render_images(
		&mut self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		sky_projview: Mat4,
	) -> crate::Result<(Arc<ImageView>, Arc<ImageView>)>
	{
		let extent2 = self.window.dimensions();
		let extent = [extent2[0], extent2[1], 1];
		if Some(extent) != self.color_image.as_ref().map(|view| view.image().extent()) {
			let color_create_info = ImageCreateInfo {
				format: Format::R16G16B16A16_SFLOAT,
				extent,
				usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
				..Default::default()
			};
			let alloc_info = AllocationCreateInfo::default();
			let color_image = Image::new(self.memory_allocator.clone(), color_create_info, alloc_info.clone())?;
			let color = ImageView::new_default(color_image)?;
			self.color_image = Some(color.clone());

			let depth_create_info = ImageCreateInfo {
				format: self.depth_stencil_format,
				extent,
				usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
				..Default::default()
			};
			let depth_image = Image::new(self.memory_allocator.clone(), depth_create_info, alloc_info)?;
			self.depth_image = Some(ImageView::new_default(depth_image)?);

			let set_layout = self.gamma_pipeline.layout().set_layouts()[0].clone();
			let set_write = [WriteDescriptorSet::image_view(0, color)];
			let set = PersistentDescriptorSet::new(&self.single_set_allocator, set_layout, set_write, [])?;
			self.color_set = Some(set);
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
				.draw(8, 2, 0, 0)?
				.end_rendering()?;
		}

		Ok((color_image, self.depth_image.clone().unwrap()))
	}

	fn present_image(&mut self, mut cb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> crate::Result<()>
	{
		if let Some((swapchain_image, acquire_future)) = self.window.get_next_image()? {
			let color_image = self.color_image.as_ref().unwrap().image().clone();

			if self.window.color_space() == ColorSpace::SrgbNonLinear {
				// perform gamma correction
				let image_extent = color_image.extent();
				let workgroups_x = image_extent[0].div_ceil(64);
				let layout = self.gamma_pipeline.layout().clone();
				cb.bind_pipeline_compute(self.gamma_pipeline.clone())?
					.bind_descriptor_sets(PipelineBindPoint::Compute, layout, 0, self.color_set.clone().unwrap())?
					.dispatch([workgroups_x, image_extent[1], 1])?;
			}

			cb.blit_image(BlitImageInfo::images(color_image, swapchain_image))?;

			let built_cb = cb.build()?;
			let transfer_future = self.transfer_manager.take_transfer_future()?;

			// submit the built command buffer, presenting it if possible
			self.window.present(built_cb, acquire_future, transfer_future)?;
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

	pub fn reset_window_min_inner_size(&self)
	{
		self.window.reset_min_inner_size()
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

fn create_sky_pipeline(device: Arc<Device>) -> crate::Result<Arc<GraphicsPipeline>>
{
	let cubemap_sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear_no_mipmap())?;
	let tex_binding = DescriptorSetLayoutBinding {
		stages: ShaderStages::FRAGMENT,
		immutable_samplers: vec![cubemap_sampler],
		..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
	};
	let set_layout_info = DescriptorSetLayoutCreateInfo {
		bindings: [(0, tex_binding)].into(),
		..Default::default()
	};
	let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

	let pipeline_layout_info = PipelineLayoutCreateInfo {
		set_layouts: vec![set_layout.clone()],
		push_constant_ranges: vec![PushConstantRange {
			stages: ShaderStages::VERTEX,
			offset: 0,
			size: std::mem::size_of::<Mat4>().try_into().unwrap(),
		}],
		..Default::default()
	};
	let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_info)?;

	let input_assembly_state = InputAssemblyState {
		topology: PrimitiveTopology::TriangleFan,
		..Default::default()
	};
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
		input_assembly_state: Some(input_assembly_state),
		viewport_state: Some(Default::default()),
		rasterization_state: Some(Default::default()),
		multisample_state: Some(Default::default()),
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
		src: r"
			#version 460

			layout(push_constant) uniform pc
			{
				mat4 sky_projview;
			};

			// Sky cube, consisting of two fans with the center being opposite corners of the cube.
			// Relative to camera at default state, -X is left, +Y is forward, and +Z is up.
			const vec3 POSITIONS[8] = {
				{ -1.0, -1.0, -1.0 },
				{ -1.0, -1.0, 1.0 },
				{ 1.0, -1.0, 1.0 },
				{ 1.0, -1.0, -1.0 },
				{ 1.0, 1.0, -1.0 },
				{ -1.0, 1.0, -1.0 },
				{ -1.0, 1.0, 1.0 },
				{ 1.0, 1.0, 1.0 },
			};
			const int INDICES[2][8] = {
				{ 0, 1, 2, 3, 4, 5, 6, 1 },
				{ 7, 1, 2, 3, 4, 5, 6, 1 },
			};

			layout(location = 0) out vec3 cube_pos; // give original vertex position to fragment shader

			void main()
			{
				int index = INDICES[gl_InstanceIndex][gl_VertexIndex];
				cube_pos = POSITIONS[index];
				vec4 new_pos = sky_projview * vec4(cube_pos, 1.0);
				gl_Position = new_pos.xyww;
			}
		",
	}
}
mod sky_fs
{
	vulkano_shaders::shader! {
		ty: "fragment",
		src: r"
			#version 460

			layout(binding = 0) uniform samplerCube sky_tex;

			layout(location = 0) in vec3 cube_pos;
			layout(location = 0) out vec4 color_out;

			void main()
			{
				color_out = texture(sky_tex, cube_pos.xzy);
			}
		",
	}
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

//
/* Texture stuff */
//
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
	(submit_transfers, model::draw_workload, draw_ui, submit_frame).into_workload()
}
fn submit_transfers(mut render_ctx: UniqueViewMut<RenderContext>) -> crate::Result<()>
{
	render_ctx.transfer_manager.submit_transfers()
}
fn draw_ui(render_ctx: UniqueView<RenderContext>, mut canvas: UniqueViewMut<Canvas>) -> crate::Result<()>
{
	canvas.draw(&render_ctx)
}

// Submit all the command buffers for this frame to present the results.
fn submit_frame(
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

	light_manager.execute_shadow_rendering(&mut cb_builder)?;

	// draw the skybox to clear the color image, and get the images
	let (color_image, depth_image) = render_ctx.get_render_images(&mut cb_builder, camera_manager.sky_projview())?;

	// opaque 3D objects
	mesh_manager.execute_rendering(&mut cb_builder, color_image.clone(), depth_image.clone())?;

	// transparent 3D objects (OIT)
	let memory_allocator = render_ctx.memory_allocator.clone();
	if let Some(transparency_renderer) = &mut render_ctx.transparency_renderer {
		transparency_renderer.process_transparency(&mut cb_builder, color_image.clone(), depth_image, memory_allocator)?;
	}

	// UI
	canvas.execute_rendering(&mut cb_builder, color_image)?;

	render_ctx.present_image(cb_builder)
}

fn check_direct_buffer_write(physical_device: &Arc<PhysicalDevice>) -> bool
{
	match physical_device.properties().device_type {
		PhysicalDeviceType::IntegratedGpu => true, // Always possible for integrated GPU.
		PhysicalDeviceType::DiscreteGpu => {
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
		_ => unreachable!(),
	}
}
