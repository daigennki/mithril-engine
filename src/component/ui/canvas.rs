/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use crate::rendercontext;
//use shipyard::EntityId;

pub struct Canvas
{
	//children: std::collections::LinkedList<EntityId>,
	projection: glam::Mat4
}
impl Canvas
{
	pub fn new(render_ctx: &mut rendercontext::RenderContext, width: u32, height: u32) 
		-> Result<Canvas, Box<dyn std::error::Error>>
	{
		let projection = glam::Mat4::orthographic_lh(0.0, width as f32, 0.0, height as f32, 0.0, 1.0);

		//let mut children = std::collections::LinkedList::<EntityId>::new();

		Ok(Canvas{
			//children: children,
			projection: projection
		})
	}

	pub fn projection(&self) -> glam::Mat4
	{
		self.projection
	}

	/*pub fn draw(&self, render_ctx: &mut rendercontext::RenderContext) -> Result<(), Box<dyn std::error::Error>>
	{
		for element in &self.elements {
			element.draw(render_ctx)?;
		}
		Ok(())
	}*/
}