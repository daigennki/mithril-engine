/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod image;

use super::rendercontext;

pub struct Canvas
{
	test_image: image::Image
}
impl Canvas
{
	pub fn new(render_context: &mut rendercontext::RenderContext) -> Result<Canvas, Box<dyn std::error::Error>>
	{
		Ok(Canvas{
			test_image: image::Image::new(render_context, std::path::Path::new("test_image.png"))?
		})
	}
}
