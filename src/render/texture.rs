/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::path::Path;
use vulkano::image::{ ImmutableImage, ImageDimensions, MipmapsCount, view::ImageView };
use vulkano::format::Format;
use vulkano::command_buffer::{ CommandBufferExecFuture, PrimaryAutoCommandBuffer };
use vulkano::sync::NowFuture;
use ddsfile::DxgiFormat;

pub struct Texture
{
	view: Arc<ImageView<ImmutableImage>>,
	dimensions: ImageDimensions
}
impl Texture
{
	pub fn new(queue: Arc<vulkano::device::Queue>, filename: &Path) 
		-> Result<(Texture, CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>), Box<dyn std::error::Error>>
	{
		// TODO: animated textures using APNG or multi-layer DDS
		let file_ext = filename.extension().ok_or("Could not determine texture file extension!")?.to_str();
		let filename_str = &filename.display().to_string();
		let (vk_fmt, dim, mip, img_raw) = match file_ext {
			Some("dds") => load_dds(filename_str)?,
			_ => load_other_format(filename_str)?
		};

		let (vk_img, upload_future) = ImmutableImage::from_iter(img_raw, dim, mip, vk_fmt, queue)?;

		Ok((
			Texture{
				view: ImageView::new(vk_img)?,
				dimensions: dim
			},
			upload_future
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

fn load_dds(filename: &str) -> Result<(Format, ImageDimensions, MipmapsCount, Vec<u8>), Box<dyn std::error::Error>>
{
	let dds_file = std::fs::File::open(filename)
		.or_else(|e| Err(format!("Could not open '{}': {}", filename, e)))?;

	let dds = ddsfile::Dds::read(dds_file)?;
	let dds_format = dds.get_dxgi_format()
		.ok_or("Could not determine DDS image format! Make sure it's in DXGI format.")?;

	let vk_fmt = dxgi_to_vulkan_format(dds_format)?;
	let dim = ImageDimensions::Dim2d{ width: dds.get_width(), height: dds.get_height(), array_layers: 1 };
	let mip = MipmapsCount::Specific(dds.get_num_mipmap_levels());
	let img_raw = dds.data;

	Ok((vk_fmt, dim, mip, img_raw))
}

fn load_other_format(filename: &str) -> Result<(Format, ImageDimensions, MipmapsCount, Vec<u8>), Box<dyn std::error::Error>>
{
	let img = image::io::Reader::open(filename)?.decode()?;

	let vk_fmt = Format::R8G8B8A8_SRGB;	// TODO: other formats such as greyscale
	let dim = ImageDimensions::Dim2d{ width: img.width(), height: img.height(), array_layers: 1 };
	let mip = MipmapsCount::One;
	let img_raw = img.into_rgba8().into_raw();

	Ok((vk_fmt, dim, mip, img_raw))
}

fn dxgi_to_vulkan_format(dxgi_format: DxgiFormat) -> Result<Format, Box<dyn std::error::Error>>
{
	match dxgi_format {
		DxgiFormat::BC1_UNorm_sRGB => Ok(Format::BC1_RGBA_SRGB_BLOCK),
		DxgiFormat::BC2_UNorm_sRGB => Ok(Format::BC2_SRGB_BLOCK),
		DxgiFormat::BC3_UNorm_sRGB => Ok(Format::BC3_SRGB_BLOCK),
		DxgiFormat::BC4_UNorm => Ok(Format::BC4_UNORM_BLOCK),
		DxgiFormat::BC5_UNorm => Ok(Format::BC5_UNORM_BLOCK),
		_ => Err("Unsupported DDS format!".into())
	}
}
