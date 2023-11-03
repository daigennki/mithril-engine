/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use ddsfile::DxgiFormat;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{BufferImageCopy, CopyBufferToImageInfo};
use vulkano::format::Format;
use vulkano::image::{
	view::{ImageView, ImageViewCreateInfo, ImageViewType}, Image, ImageCreateFlags, ImageCreateInfo, ImageSubresourceLayers,
	ImageUsage,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

use crate::GenericEngineError;

pub struct Texture
{
	view: Arc<ImageView>,
	dimensions: [u32; 2],
}
impl Texture
{
	pub fn new(
		memory_allocator: Arc<StandardMemoryAllocator>,
		path: &Path,
	) -> Result<(Self, CopyBufferToImageInfo), GenericEngineError>
	{
		// TODO: animated textures using APNG, animated JPEG-XL, or multi-layer DDS
		log::info!("Loading texture file '{}'...", path.display());
		let file_ext = path
			.extension()
			.ok_or("Could not determine texture file extension!")?
			.to_str();
		let (vk_fmt, dim, mip, img_raw) = match file_ext {
			Some("dds") => load_dds(path)?,
			_ => load_other_format(path)?,
		};

		Self::new_from_iter(memory_allocator, img_raw, vk_fmt, dim, mip)
	}

	pub fn new_from_iter<Px, I>(
		memory_allocator: Arc<StandardMemoryAllocator>,
		iter: I,
		format: Format,
		dimensions: [u32; 2],
		mip_levels: u32,
	) -> Result<(Self, CopyBufferToImageInfo), GenericEngineError>
	where
		Px: Send + Sync + bytemuck::Pod,
		[Px]: vulkano::buffer::BufferContents,
		I: IntoIterator<Item = Px>,
		I::IntoIter: ExactSizeIterator,
	{
		let buffer_info = BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() };
		let buf_allocation_info = AllocationCreateInfo {
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buf = Buffer::from_iter(memory_allocator.clone(), buffer_info, buf_allocation_info, iter)?;

		let image_info = ImageCreateInfo {
			format,
			extent: [ dimensions[0], dimensions[1], 1 ],
			mip_levels,
			usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
			..Default::default()
		};
		let image = Image::new(memory_allocator, image_info, AllocationCreateInfo::default())?;

		let view = ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image))?;

		// generate copies for every mipmap level
		let mut regions = Vec::with_capacity(mip_levels as usize);
		let mut mip_width = dimensions[0];
		let mut mip_height = dimensions[1];
		let mut buffer_offset: u64 = 0;
		for mip_level in 0..mip_levels {
			regions.push(BufferImageCopy {
				buffer_offset,
				image_subresource: ImageSubresourceLayers {
					mip_level,
					..ImageSubresourceLayers::from_parameters(format, 1)
				},
				image_extent: [ mip_width, mip_height, 1],
				..Default::default()
			});
			
			let block_extent = format.block_extent();
			let block_size = format.block_size();
			let x_blocks = mip_width.div_ceil(block_extent[0]) as u64;
			let y_blocks = mip_height.div_ceil(block_extent[1]) as u64;
			let mip_size = x_blocks * y_blocks * block_size;
			buffer_offset += mip_size;

			mip_width /= 2;
			mip_height /= 2;
		}

		let copy_to_image = CopyBufferToImageInfo{
			regions: regions.into(),
			..CopyBufferToImageInfo::buffer_image(staging_buf, image)
		};

		Ok((Texture { view, dimensions }, copy_to_image))
	}

	pub fn view(&self) -> Arc<ImageView>
	{
		self.view.clone()
	}

	pub fn dimensions(&self) -> [u32; 2] 
	{
		self.dimensions
	}
}

pub struct CubemapTexture
{
	view: Arc<ImageView>,
	dimensions: [u32; 2],
}
impl CubemapTexture
{
	/// `faces` is paths to textures of each face of the cubemap, in order of +X, -X, +Y, -Y, +Z, -Z
	pub fn new(
		memory_allocator: Arc<StandardMemoryAllocator>,
		faces: [PathBuf; 6],
	) -> Result<(Self, CopyBufferToImageInfo), GenericEngineError>
	{
		let mut combined_data = Vec::<u8>::new();
		let mut cube_fmt = None;
		let mut cube_dim = None;

		for face in faces {
			log::info!("Loading texture file '{}'...", face.display());

			let file_ext = face
				.extension()
				.ok_or("Could not determine texture file extension!")?
				.to_str();
			let (vk_fmt, dim, mip, img_raw) = match file_ext {
				Some("dds") => load_dds(&face)?,
				_ => load_other_format(&face)?,
			};

			// TODO: ignore other mipmap levels, if there are any
			if mip > 1 {
				return Err(format!(
					"expected texture file with only one mipmap level, got {} mipmap levels",
					mip	
				)
				.into());
			}

			if let Some(f) = cube_fmt.as_ref() {
				if *f != vk_fmt {
					return Err("not all faces of a cubemap have the same format!".into());
				}
			} else {
				cube_fmt = Some(vk_fmt);
			}

			if let Some(d) = cube_dim.as_ref() {
				if *d != dim {
					return Err("not all faces of a cubemap have the same dimensions!".into());
				}
			} else {
				cube_dim = Some(dim);
			}

			combined_data.extend(img_raw);
		}

		Self::new_from_iter(
			memory_allocator,
			combined_data,
			cube_fmt.unwrap(),
			cube_dim.unwrap(),
			1,
		)
	}
	pub fn new_from_iter<Px, I>(
		memory_allocator: Arc<StandardMemoryAllocator>,
		iter: I,
		vk_fmt: Format,
		dimensions: [u32; 2],
		mip: u32,
	) -> Result<(Self, CopyBufferToImageInfo), GenericEngineError>
	where
		Px: Send + Sync + bytemuck::Pod,
		[Px]: vulkano::buffer::BufferContents,
		I: IntoIterator<Item = Px>,
		I::IntoIter: ExactSizeIterator,
	{
		let (vk_img, staging_info) = create_cubemap_image(iter, dimensions, mip, vk_fmt, memory_allocator)?;
		let mut view_create_info = vulkano::image::view::ImageViewCreateInfo::from_image(&vk_img);
		view_create_info.view_type = ImageViewType::Cube;

		Ok((
			CubemapTexture {
				view: ImageView::new(vk_img, view_create_info)?,
				dimensions,
			},
			staging_info,
		))
	}

	pub fn view(&self) -> Arc<ImageView>
	{
		self.view.clone()
	}

	pub fn dimensions(&self) -> [u32; 2] 
	{
		self.dimensions
	}
}

fn create_cubemap_image<Px, I>(
	iter: I,
	dimensions: [u32; 2],
	mip_levels: u32,
	format: Format,
	allocator: Arc<StandardMemoryAllocator>,
) -> Result<(Arc<Image>, CopyBufferToImageInfo), GenericEngineError>
where
	Px: Send + Sync + bytemuck::Pod,
	[Px]: vulkano::buffer::BufferContents,
	I: IntoIterator<Item = Px>,
	I::IntoIter: ExactSizeIterator,
{
	let buffer_info = BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() };
	let buf_allocation_info = AllocationCreateInfo {
		memory_type_filter: MemoryTypeFilter::PREFER_HOST |  MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
		..Default::default()
	};
	let src = Buffer::from_iter(allocator.clone(), buffer_info, buf_allocation_info, iter)?;

	let image_info = ImageCreateInfo {
		flags: ImageCreateFlags::CUBE_COMPATIBLE,
		format,
		extent: [ dimensions[0], dimensions[1], 1 ],
		array_layers: 6,
		mip_levels,
		usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
		..Default::default()
	};
	let image = Image::new(allocator, image_info, AllocationCreateInfo::default())?;

	Ok((image.clone(), CopyBufferToImageInfo::buffer_image(src, image)))
}

fn load_dds(path: &Path) -> Result<(Format, [u32; 2], u32, Vec<u8>), GenericEngineError>
{
	let dds_file = std::fs::File::open(path).or_else(|e| Err(format!("Could not open '{}': {}", path.display(), e)))?;

	let dds = ddsfile::Dds::read(dds_file)?;
	let dds_format = dds
		.get_dxgi_format()
		.ok_or("Could not determine DDS image format! Make sure it's in DXGI format.")?;

	let vk_fmt = dxgi_to_vulkan_format(dds_format)?;
	let dim = [ dds.get_width(), dds.get_height() ];
	let mip = dds.get_num_mipmap_levels();
	let img_raw = dds.data;

	Ok((vk_fmt, dim, mip, img_raw))
}

fn load_other_format(path: &Path) -> Result<(Format, [u32; 2], u32, Vec<u8>), GenericEngineError>
{
	let img = image::io::Reader::open(path)?.decode()?;

	let vk_fmt = Format::R8G8B8A8_SRGB; // TODO: other formats such as greyscale
	let dim = [ img.width(), img.height() ];
	let img_raw = img.into_rgba8().into_raw();

	Ok((vk_fmt, dim, 1, img_raw))
}

fn dxgi_to_vulkan_format(dxgi_format: DxgiFormat) -> Result<Format, GenericEngineError>
{
	Ok(match dxgi_format {
		DxgiFormat::BC1_UNorm_sRGB => Format::BC1_RGBA_SRGB_BLOCK,
		DxgiFormat::BC2_UNorm_sRGB => Format::BC2_SRGB_BLOCK,
		DxgiFormat::BC3_UNorm_sRGB => Format::BC3_SRGB_BLOCK,
		DxgiFormat::BC4_UNorm => Format::BC4_UNORM_BLOCK,
		DxgiFormat::BC5_UNorm => Format::BC5_UNORM_BLOCK,
		_ => return Err("Unsupported DDS format!".into()),
	})
}
