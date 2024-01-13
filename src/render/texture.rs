/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use ddsfile::DxgiFormat;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::format::Format;
use vulkano::image::{
	view::{ImageView, ImageViewCreateInfo, ImageViewType},
	ImageCreateFlags, ImageCreateInfo, ImageUsage,
};
use vulkano::DeviceSize;

use super::RenderContext;
use crate::EngineError;

/// Load an image file as a 2D texture into memory.
///
/// The results of this are cached; if the image was already loaded, it'll use the loaded texture.
pub fn new(render_ctx: &mut RenderContext, path: &Path) -> crate::Result<Arc<ImageView>>
{
	if let Some(tex) = render_ctx.textures.get(path) {
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
	let image = render_ctx.new_image(img_raw, image_info)?;
	let view = ImageView::new_default(image)?;

	render_ctx.textures.insert(path.to_path_buf(), view.clone());

	Ok(view)
}

/// Load six image files as cubemap textures into memory.
///
/// `faces` is paths to textures of each face of the cubemap, in order of +X, -X, +Y, -Y, +Z, -Z.
///
/// Unlike `new`, the results of this are *not* cached.
pub fn new_cubemap(render_ctx: &mut RenderContext, faces: [PathBuf; 6]) -> crate::Result<Arc<ImageView>>
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
	let image = render_ctx.new_image(combined_data, image_info)?;

	let view_create_info = ImageViewCreateInfo {
		view_type: ImageViewType::Cube,
		..ImageViewCreateInfo::from_image(&image)
	};
	Ok(ImageView::new(image, view_create_info)?)
}

fn load_texture(path: &Path) -> crate::Result<(Format, [u32; 2], u32, Vec<u8>)>
{
	log::info!("Loading texture file '{}'...", path.display());

	let file_ext = path
		.extension()
		.ok_or("Could not determine texture file extension!")?
		.to_str();

	match file_ext {
		Some("dds") => load_dds(path),
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

	// BC7_UNorm is treated as sRGB for now since Compressonator doesn't support converting to
	// BC7_UNorm_sRGB, even though the data itself appears to be in sRGB gamma.
	let vk_fmt = match dds_format {
		DxgiFormat::BC1_UNorm_sRGB => Format::BC1_RGBA_SRGB_BLOCK,
		DxgiFormat::BC4_UNorm => Format::BC4_UNORM_BLOCK,
		DxgiFormat::BC5_UNorm => Format::BC5_UNORM_BLOCK,
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

pub fn get_mip_size(format: Format, mip_width: u32, mip_height: u32) -> DeviceSize
{
	let block_extent = format.block_extent();
	let block_size = format.block_size();
	let x_blocks = mip_width.div_ceil(block_extent[0]) as DeviceSize;
	let y_blocks = mip_height.div_ceil(block_extent[1]) as DeviceSize;
	x_blocks * y_blocks * block_size
}
