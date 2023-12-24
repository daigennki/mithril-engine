/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use ddsfile::DxgiFormat;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::{allocator::SubbufferAllocator, BufferContents, Subbuffer};
use vulkano::command_buffer::{BufferImageCopy, CopyBufferToImageInfo};
use vulkano::format::Format;
use vulkano::image::{
	view::{ImageView, ImageViewCreateInfo, ImageViewType},
	Image, ImageCreateFlags, ImageCreateInfo, ImageSubresourceLayers, ImageUsage,
};
use vulkano::memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryAllocatorError, StandardMemoryAllocator};
use vulkano::DeviceSize;

use crate::EngineError;

pub struct Texture
{
	view: Arc<ImageView>,
}
impl Texture
{
	pub fn new(
		memory_allocator: Arc<StandardMemoryAllocator>,
		subbuffer_allocator: &mut SubbufferAllocator,
		path: &Path,
	) -> crate::Result<(Self, CopyBufferToImageInfo)>
	{
		// TODO: animated textures using APNG, animated JPEG-XL, or multi-layer DDS
		let (vk_fmt, dim, mip_count, img_raw) = load_texture(path)?;

		Self::new_from_slice(
			memory_allocator,
			subbuffer_allocator,
			img_raw.as_slice(),
			vk_fmt,
			dim,
			mip_count,
			1,
		)
	}

	pub fn new_from_slice<Px>(
		memory_allocator: Arc<StandardMemoryAllocator>,
		subbuffer_allocator: &mut SubbufferAllocator,
		data: &[Px],
		format: Format,
		dimensions: [u32; 2],
		mip_levels: u32,
		array_layers: u32,
	) -> crate::Result<(Self, CopyBufferToImageInfo)>
	where
		Px: BufferContents + Copy,
	{
		let staging_buf = get_tex_staging_buf(subbuffer_allocator, data, format)?;

		let image_info = ImageCreateInfo {
			format,
			extent: [dimensions[0], dimensions[1], 1],
			array_layers,
			mip_levels,
			usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
			..Default::default()
		};
		let image = Image::new(memory_allocator, image_info, AllocationCreateInfo::default())
			.map_err(|e| EngineError::vulkan_error("failed to create image", e))?;

		let view = ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image))
			.map_err(|e| EngineError::vulkan_error("failed to create image view", e))?;

		// generate copies for every mipmap level
		let mut regions = Vec::with_capacity(mip_levels as usize);
		let mut mip_width = dimensions[0];
		let mut mip_height = dimensions[1];
		let mut buffer_offset: DeviceSize = 0;
		for mip_level in 0..mip_levels {
			regions.push(BufferImageCopy {
				buffer_offset,
				image_subresource: ImageSubresourceLayers {
					mip_level,
					..ImageSubresourceLayers::from_parameters(format, array_layers)
				},
				image_extent: [mip_width, mip_height, 1],
				..Default::default()
			});

			buffer_offset += get_mip_size(format, mip_width, mip_height);
			mip_width /= 2;
			mip_height /= 2;
		}

		let copy_to_image = CopyBufferToImageInfo {
			regions: regions.into(),
			..CopyBufferToImageInfo::buffer_image(staging_buf, image)
		};

		Ok((Texture { view }, copy_to_image))
	}

	pub fn view(&self) -> &Arc<ImageView>
	{
		&self.view
	}

	pub fn dimensions(&self) -> [u32; 2]
	{
		let extent = self.view.image().extent();
		[extent[0], extent[1]]
	}
}

pub struct CubemapTexture
{
	view: Arc<ImageView>,
}
impl CubemapTexture
{
	/// `faces` is paths to textures of each face of the cubemap, in order of +X, -X, +Y, -Y, +Z, -Z
	pub fn new(
		memory_allocator: Arc<StandardMemoryAllocator>,
		subbuffer_allocator: &mut SubbufferAllocator,
		faces: [PathBuf; 6],
	) -> crate::Result<(Self, CopyBufferToImageInfo)>
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

		Self::new_from_slice(
			memory_allocator,
			subbuffer_allocator,
			combined_data.as_slice(),
			cube_fmt.unwrap(),
			cube_dim.unwrap(),
		)
	}

	pub fn new_from_slice<Px>(
		memory_allocator: Arc<StandardMemoryAllocator>,
		subbuffer_allocator: &mut SubbufferAllocator,
		data: &[Px],
		format: Format,
		dimensions: [u32; 2],
	) -> crate::Result<(Self, CopyBufferToImageInfo)>
	where
		Px: BufferContents + Copy,
	{
		let staging_buf = get_tex_staging_buf(subbuffer_allocator, data, format)?;

		let image_info = ImageCreateInfo {
			flags: ImageCreateFlags::CUBE_COMPATIBLE,
			format,
			extent: [dimensions[0], dimensions[1], 1],
			array_layers: 6,
			usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
			..Default::default()
		};
		let image = Image::new(memory_allocator, image_info, AllocationCreateInfo::default())
			.map_err(|e| EngineError::vulkan_error("failed to create image", e))?;

		let view_create_info = ImageViewCreateInfo {
			view_type: ImageViewType::Cube,
			..ImageViewCreateInfo::from_image(&image)
		};
		let view = ImageView::new(image.clone(), view_create_info)
			.map_err(|e| EngineError::vulkan_error("failed to create image view", e))?;

		let copy_to_image = CopyBufferToImageInfo::buffer_image(staging_buf, image);

		Ok((CubemapTexture { view }, copy_to_image))
	}

	pub fn view(&self) -> &Arc<ImageView>
	{
		&self.view
	}
}

fn get_tex_staging_buf<Px>(
	subbuffer_allocator: &mut SubbufferAllocator,
	data: &[Px],
	format: Format,
) -> crate::Result<Subbuffer<[Px]>>
where
	Px: BufferContents + Copy,
{
	// We allocate a subbuffer using a `DeviceLayout` here so that it's aligned to the block size
	// of the format.
	let data_size_bytes = (data.len() * std::mem::size_of::<Px>()).try_into().unwrap();
	let device_layout = DeviceLayout::from_size_alignment(data_size_bytes, format.block_size())
		.ok_or("Texture::new_from_slice: slice is empty or alignment is not a power of two")?;

	let staging_buf: Subbuffer<[Px]> = subbuffer_allocator
		.allocate(device_layout)
		.map_err(|e| match e {
			MemoryAllocatorError::AllocateDeviceMemory(validated) => {
				EngineError::vulkan_error("failed to allocate staging buffer", validated)
			}
			other => EngineError::new("failed to allocate staging buffer", other),
		})?
		.reinterpret();

	staging_buf.write().unwrap().copy_from_slice(data);

	Ok(staging_buf)
}

fn load_texture(path: &Path) -> crate::Result<(Format, [u32; 2], u32, Vec<u8>)>
{
	log::info!("Loading texture file '{}'...", path.display());

	let file_ext = path
		.extension()
		.ok_or("Could not determine texture file extension!")?
		.to_str();

	match file_ext {
		Some("dds") => Ok(load_dds(path)?),
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
fn load_dds(path: &Path) -> crate::Result<(Format, [u32; 2], u32, Vec<u8>)>
{
	let dds_file = std::fs::File::open(path).map_err(|e| EngineError::new("couldn't open DDS file", e))?;

	let dds = ddsfile::Dds::read(dds_file).map_err(|e| EngineError::new("failed to read DDS file", e))?;
	let dds_format = dds
		.get_dxgi_format()
		.ok_or("Could not determine DDS image format! Make sure it has a DXGI format.")?;

	let vk_fmt = match dds_format {
		DxgiFormat::BC1_UNorm_sRGB => Format::BC1_RGBA_SRGB_BLOCK,
		DxgiFormat::BC4_UNorm => Format::BC4_UNORM_BLOCK,
		DxgiFormat::BC5_UNorm => Format::BC5_UNORM_BLOCK,
		// treat BC7_UNorm as sRGB for now since Compressonator doesn't support converting to BC7_UNorm_sRGB,
		// even though the data itself appears to be in sRGB gamma
		DxgiFormat::BC7_UNorm => Format::BC7_SRGB_BLOCK,
		DxgiFormat::BC7_UNorm_sRGB => Format::BC7_SRGB_BLOCK,
		format => {
			let e = UnsupportedDdsFormat { format };
			return Err(EngineError::new("failed to read DDS file", e));
		}
	};
	let dim = [dds.get_width(), dds.get_height()];
	let mip_count = dds.get_num_mipmap_levels();

	Ok((vk_fmt, dim, mip_count, dds.data))
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
