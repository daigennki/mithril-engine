/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use ddsfile::DxgiFormat;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::CopyBufferToImageInfo;
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::image::{
	view::ImageView, view::ImageViewCreateInfo, view::ImageViewType, ImageCreateFlags, ImageDimensions, ImageLayout,
	ImageUsage, ImmutableImage, MipmapsCount,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};

use crate::GenericEngineError;

pub struct Texture
{
	view: Arc<ImageView<ImmutableImage>>,
	dimensions: ImageDimensions,
}
impl Texture
{
	pub fn new(
		memory_allocator: &StandardMemoryAllocator,
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
		memory_allocator: &StandardMemoryAllocator,
		iter: I,
		vk_fmt: Format,
		dimensions: ImageDimensions,
		mip: MipmapsCount,
	) -> Result<(Self, CopyBufferToImageInfo), GenericEngineError>
	where
		Px: Send + Sync + bytemuck::Pod,
		[Px]: vulkano::buffer::BufferContents,
		I: IntoIterator<Item = Px>,
		I::IntoIter: ExactSizeIterator,
	{
		let device = memory_allocator.device().clone();

		let dst_img_usage = ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED;
		let buffer_info = BufferCreateInfo {
			usage: BufferUsage::TRANSFER_SRC,
			..Default::default()
		};
		let buf_allocation_info = AllocationCreateInfo {
			usage: MemoryUsage::Upload,
			..Default::default()
		};
		let staging_buf = Buffer::from_iter(memory_allocator, buffer_info, buf_allocation_info, iter)?;
		let queue_families: Vec<_> = device.active_queue_family_indices().into();
		let (dst_img, initializer) = ImmutableImage::uninitialized(
			memory_allocator,
			dimensions,
			vk_fmt,
			mip,
			dst_img_usage,
			ImageCreateFlags::empty(),
			ImageLayout::ShaderReadOnlyOptimal,
			queue_families,
		)?;

		let view = ImageView::new(dst_img.clone(), ImageViewCreateInfo::from_image(&dst_img))?;

		// TODO: also copy mipmaps

		Ok((
			Texture { view, dimensions },
			CopyBufferToImageInfo::buffer_image(staging_buf, initializer),
		))
	}

	pub fn view(&self) -> Arc<ImageView<ImmutableImage>>
	{
		self.view.clone()
	}

	pub fn dimensions(&self) -> ImageDimensions
	{
		self.dimensions
	}
}

pub struct CubemapTexture
{
	view: Arc<ImageView<ImmutableImage>>,
	dimensions: ImageDimensions,
}
impl CubemapTexture
{
	/// `faces` is paths to textures of each face of the cubemap, in order of +X, -X, +Y, -Y, +Z, -Z
	pub fn new(
		memory_allocator: &StandardMemoryAllocator,
		faces: [PathBuf; 6],
	) -> Result<(Self, CopyBufferToImageInfo), GenericEngineError>
	{
		// TODO: animated textures using APNG or multi-layer DDS

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
			if let MipmapsCount::Specific(count) = mip {
				return Err(format!(
					"expected texture file with only one mipmap level, got {} mipmap levels",
					count
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

		if let ImageDimensions::Dim2d { array_layers, .. } = cube_dim.as_mut().unwrap() {
			*array_layers = 6;
		}

		Self::new_from_iter(
			memory_allocator,
			combined_data,
			cube_fmt.unwrap(),
			cube_dim.unwrap(),
			MipmapsCount::One,
		)
	}
	pub fn new_from_iter<Px, I>(
		memory_allocator: &StandardMemoryAllocator,
		iter: I,
		vk_fmt: Format,
		dimensions: ImageDimensions,
		mip: MipmapsCount,
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

	pub fn view(&self) -> Arc<ImageView<ImmutableImage>>
	{
		self.view.clone()
	}

	pub fn dimensions(&self) -> ImageDimensions
	{
		self.dimensions
	}
}

fn create_cubemap_image<Px, I>(
	iter: I,
	dimensions: ImageDimensions,
	mip_levels: MipmapsCount,
	format: Format,
	allocator: &StandardMemoryAllocator,
) -> Result<(Arc<ImmutableImage>, CopyBufferToImageInfo), GenericEngineError>
where
	Px: Send + Sync + bytemuck::Pod,
	[Px]: vulkano::buffer::BufferContents,
	I: IntoIterator<Item = Px>,
	I::IntoIter: ExactSizeIterator,
{
	let device = allocator.device().clone();

	let buffer_info = BufferCreateInfo {
		usage: BufferUsage::TRANSFER_SRC,
		..Default::default()
	};
	let buf_allocation_info = AllocationCreateInfo {
		usage: MemoryUsage::Upload,
		..Default::default()
	};
	let src = Buffer::from_iter(allocator, buffer_info, buf_allocation_info, iter)?;

	let usage = ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED;
	let flags = ImageCreateFlags::CUBE_COMPATIBLE;

	let queue_families: Vec<_> = device.active_queue_family_indices().into();
	let (image, initializer) = ImmutableImage::uninitialized(
		allocator,
		dimensions,
		format,
		mip_levels,
		usage,
		flags,
		ImageLayout::ShaderReadOnlyOptimal,
		queue_families,
	)?;

	Ok((image, CopyBufferToImageInfo::buffer_image(src, initializer)))
}

fn load_dds(path: &Path) -> Result<(Format, ImageDimensions, MipmapsCount, Vec<u8>), GenericEngineError>
{
	let dds_file = std::fs::File::open(path).or_else(|e| Err(format!("Could not open '{}': {}", path.display(), e)))?;

	let dds = ddsfile::Dds::read(dds_file)?;
	let dds_format = dds
		.get_dxgi_format()
		.ok_or("Could not determine DDS image format! Make sure it's in DXGI format.")?;

	let vk_fmt = dxgi_to_vulkan_format(dds_format)?;
	let dim = ImageDimensions::Dim2d {
		width: dds.get_width(),
		height: dds.get_height(),
		array_layers: 1,
	};
	let mip = MipmapsCount::Specific(dds.get_num_mipmap_levels());
	let img_raw = dds.data;

	Ok((vk_fmt, dim, mip, img_raw))
}

fn load_other_format(path: &Path) -> Result<(Format, ImageDimensions, MipmapsCount, Vec<u8>), GenericEngineError>
{
	let img = image::io::Reader::open(path)?.decode()?;

	let vk_fmt = Format::R8G8B8A8_SRGB; // TODO: other formats such as greyscale
	let dim = ImageDimensions::Dim2d {
		width: img.width(),
		height: img.height(),
		array_layers: 1,
	};
	let mip = MipmapsCount::One;
	let img_raw = img.into_rgba8().into_raw();

	Ok((vk_fmt, dim, mip, img_raw))
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
