/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use shipyard::EntityId;

pub struct Canvas
{
	children: std::collections::LinkedList<EntityId>,
	projection: glam::Mat4
}
impl Canvas
{
	pub fn new(width: u32, height: u32) 
		-> Result<Canvas, Box<dyn std::error::Error>>
	{
		let half_width = width as f32 / 2.0;
		let half_height = height as f32 / 2.0;
		let projection = glam::Mat4::orthographic_lh(-half_width, half_width, -half_height, half_height, 0.0, 1.0);

		Ok(Canvas{
			children: std::collections::LinkedList::<EntityId>::new(),
			projection: projection
		})
	}

	pub fn projection(&self) -> glam::Mat4
	{
		self.projection
	}

	pub fn add_child(&mut self, eid: EntityId)
	{
		self.children.push_back(eid);
	}
}