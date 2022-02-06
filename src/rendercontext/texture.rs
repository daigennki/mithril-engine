/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use std::path::Path;
use vulkano::image::ImmutableImage;
use vulkano::image::ImageDimensions;
use vulkano::image::MipmapsCount;
use vulkano::format::Format;
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::sync::NowFuture;
use ddsfile::DxgiFormat;

pub struct Texture
{
	vk_img: Arc<ImmutableImage>
}
impl Texture
{
	pub fn new(queue: Arc<vulkano::device::Queue>, filename: &Path) 
		-> Result<(Texture, CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>), Box<dyn std::error::Error>>
	{
		// TODO: animated textures using APNG or multi-layer DDS

		let vk_fmt;
		let dim;
		let mip;
		let img_raw;
		let file_ext = filename.extension().ok_or("Could not determine texture file extension!")?;
		if file_ext == "dds" {
			// process DDS
			let dds_file = std::fs::File::open(filename)
				.or_else(|e| Err(format!("Could not open '{}': {}", filename.display(), e)))?;

			let dds = ddsfile::Dds::read(dds_file)?;
			let dds_format = dds.get_dxgi_format()
				.ok_or("Could not determine DDS image format! Make sure it's in DXGI format.")?;

			vk_fmt = dxgi_to_vulkan_format(dds_format)?;
			dim = ImageDimensions::Dim2d{ width: dds.get_width(), height: dds.get_height(), array_layers: 1 };
			mip = MipmapsCount::Specific(dds.get_num_mipmap_levels());
			img_raw = dds.data;
		} else {
			// process other format
			let img = image::io::Reader::open(filename)?.decode()?;

			vk_fmt = Format::R8G8B8A8_SRGB;	// TODO: other formats such as greyscale
			dim = ImageDimensions::Dim2d{ width: img.width(), height: img.height(), array_layers: 1 };
			mip = MipmapsCount::One;
			img_raw = img.into_rgba8().into_raw();
		}

		let (vk_img, upload_future) = ImmutableImage::from_iter(img_raw, dim, mip, vk_fmt, queue)?;

		Ok((
			Texture{
				vk_img: vk_img
			},
			upload_future
		))
	}
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
