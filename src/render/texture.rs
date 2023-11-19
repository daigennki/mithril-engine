/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use ddsfile::DxgiFormat;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::{allocator::SubbufferAllocator, BufferContents};
use vulkano::command_buffer::{BufferImageCopy, CopyBufferToImageInfo};
use vulkano::format::Format;
use vulkano::image::{
	view::{ImageView, ImageViewCreateInfo, ImageViewType}, Image, ImageCreateFlags, ImageCreateInfo, ImageSubresourceLayers,
	ImageUsage,
};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::DeviceSize;

use crate::GenericEngineError;

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
	) -> Result<(Self, CopyBufferToImageInfo), GenericEngineError>
	{
		// TODO: animated textures using APNG, animated JPEG-XL, or multi-layer DDS
		log::info!("Loading texture file '{}'...", path.display());
		let file_ext = path
			.extension()
			.ok_or("Could not determine texture file extension!")?
			.to_str();
		let (vk_fmt, dim, mip_count, img_raw) = match file_ext {
			Some("dds") => load_dds(path)?,
			_ => load_other_format(path)?,
		};

		Self::new_from_slice(memory_allocator, subbuffer_allocator, img_raw.as_slice(), vk_fmt, dim, mip_count)
	}

	pub fn new_from_slice<Px>(
		memory_allocator: Arc<StandardMemoryAllocator>,
		subbuffer_allocator: &mut SubbufferAllocator,
		data: &[Px],
		format: Format,
		dimensions: [u32; 2],
		mip_levels: u32,
	) -> Result<(Self, CopyBufferToImageInfo), GenericEngineError>
	where
		Px: BufferContents + Copy,
	{
		let staging_buf = subbuffer_allocator.allocate_slice(data.len().try_into()?)?;
		staging_buf.write().unwrap().copy_from_slice(data);

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
		let mut buffer_offset: DeviceSize = 0;
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

			buffer_offset += get_mip_size(format, mip_width, mip_height);
			mip_width /= 2;
			mip_height /= 2;
		}

		let copy_to_image = CopyBufferToImageInfo{
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
		[ extent[0], extent[1] ]
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
			let (vk_fmt, dim, _, img_raw) = match file_ext {
				Some("dds") => load_dds(&face)?,
				_ => load_other_format(&face)?,
			};

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

			let mip_size = get_mip_size(vk_fmt, dim[0], dim[1]).try_into().unwrap();
			combined_data.extend(&img_raw[..mip_size]);
		}

		Self::new_from_slice(
			memory_allocator,
			subbuffer_allocator,
			combined_data.as_slice(),
			cube_fmt.unwrap(),
			cube_dim.unwrap(),
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
	) -> Result<(Self, CopyBufferToImageInfo), GenericEngineError>
	where
		Px: BufferContents + Copy,
	{
		let staging_buf = subbuffer_allocator.allocate_slice(data.len().try_into()?)?;
		staging_buf.write().unwrap().copy_from_slice(data);

		let image_info = ImageCreateInfo {
			flags: ImageCreateFlags::CUBE_COMPATIBLE,
			format,
			extent: [ dimensions[0], dimensions[1], 1 ],
			array_layers: 6,
			mip_levels,
			usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
			..Default::default()
		};
		let image = Image::new(memory_allocator, image_info, AllocationCreateInfo::default())?;

		let view_create_info = ImageViewCreateInfo {
			view_type: ImageViewType::Cube,
			..ImageViewCreateInfo::from_image(&image)
		};
		let view = ImageView::new(image.clone(), view_create_info)?;

		Ok((CubemapTexture { view }, CopyBufferToImageInfo::buffer_image(staging_buf, image)))
	}

	pub fn view(&self) -> &Arc<ImageView>
	{
		&self.view
	}
}

fn load_dds(path: &Path) -> Result<(Format, [u32; 2], u32, Vec<u8>), GenericEngineError>
{
	let dds_file = std::fs::File::open(path).or_else(|e| Err(format!("Could not open '{}': {}", path.display(), e)))?;

	let dds = ddsfile::Dds::read(dds_file)?;
	let dds_format = dds
		.get_dxgi_format()
		.ok_or("Could not determine DDS image format! Make sure it's in DXGI format.")?;

	let vk_fmt = match dds_format {
		DxgiFormat::BC1_UNorm_sRGB => Format::BC1_RGBA_SRGB_BLOCK,
		DxgiFormat::BC3_UNorm_sRGB => Format::BC3_SRGB_BLOCK,
		DxgiFormat::BC4_UNorm => Format::BC4_UNORM_BLOCK,
		DxgiFormat::BC5_UNorm => Format::BC5_UNORM_BLOCK,
		_ => return Err("Unsupported DDS format!".into()),
	};
	let dim = [ dds.get_width(), dds.get_height() ];
	let mip_count = dds.get_num_mipmap_levels();
	let img_raw = dds.data;

	Ok((vk_fmt, dim, mip_count, img_raw))
}

fn load_other_format(path: &Path) -> Result<(Format, [u32; 2], u32, Vec<u8>), GenericEngineError>
{
	let img = image::io::Reader::open(path)?.decode()?;

	let vk_fmt = Format::R8G8B8A8_SRGB; // TODO: other formats such as greyscale
	let dim = [ img.width(), img.height() ];
	let img_raw = img.into_rgba8().into_raw();

	Ok((vk_fmt, dim, 1, img_raw))
}

fn get_mip_size(format: Format, mip_width: u32, mip_height: u32) -> DeviceSize
{
	let block_extent = format.block_extent();
	let block_size = format.block_size();
	let x_blocks = mip_width.div_ceil(block_extent[0]) as u64;
	let y_blocks = mip_height.div_ceil(block_extent[1]) as u64;
	x_blocks * y_blocks * block_size
}
