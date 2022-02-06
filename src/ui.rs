/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use super::rendercontext;

pub struct Canvas
{
}
impl Canvas
{
	pub fn new(render_context: &mut rendercontext::RenderContext) -> Result<Canvas, Box<dyn std::error::Error>>
	{
		Ok(Canvas{
		})
	}
}