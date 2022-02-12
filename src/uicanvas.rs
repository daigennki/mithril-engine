/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use crate::rendercontext;
use crate::component::ui;

pub struct UICanvas
{
	elements: std::collections::LinkedList<Box<dyn ui::UIElement>>
}
impl UICanvas
{
	pub fn new(render_ctx: &mut rendercontext::RenderContext, width: u32, height: u32) 
		-> Result<UICanvas, Box<dyn std::error::Error>>
	{
		let projection = glam::Mat4::orthographic_lh(0.0, width as f32, 0.0, height as f32, 0.0, 1.0);

		let mut elements: std::collections::LinkedList<Box<dyn ui::UIElement>> = std::collections::LinkedList::new();
		
		let test_image = ui::img::Img::new(
			render_ctx, glam::Vec2::new(640.0, 360.0), projection, std::path::Path::new("test_image.png")
		)?;
		elements.push_back(Box::new(test_image));

		Ok(UICanvas{
			elements: elements
		})
	}

	pub fn draw(&self, render_ctx: &mut rendercontext::RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		for element in &self.elements {
			element.draw(render_ctx)?;
		}
		Ok(())
	}
}