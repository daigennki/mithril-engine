/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod quad;
pub mod img;
pub mod canvas;

use vulkano::command_buffer::DrawError;
use crate::rendercontext;

/// Common trait for UI elements.
pub trait UIElement
{
	fn draw(&self, render_ctx: &mut rendercontext::RenderContext) -> Result<(), DrawError>;
}
