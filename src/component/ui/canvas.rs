/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use glam::*;

#[derive(shipyard::Unique)]
pub struct Canvas
{
	base_dimensions: [u32; 2],
	projection: Mat4
}
impl Canvas
{
	pub fn new(canvas_width: u32, canvas_height: u32, screen_width: u32, screen_height: u32) 
		-> Result<Self, Box<dyn std::error::Error>>
	{	
		Ok(Canvas{ 
			base_dimensions: [ canvas_width, canvas_height ], 
			projection: calculate_projection(canvas_width, canvas_height, screen_width, screen_height) 
		})
	}

	/// Run this function whenever the screen resizes, to adjust the canvas aspect ratio to fit.
	pub fn on_screen_resize(&mut self, screen_width: u32, screen_height: u32)
	{
		self.projection = calculate_projection(self.base_dimensions[0], self.base_dimensions[1], screen_width, screen_height)
	}

	pub fn projection(&self) -> Mat4
	{
		self.projection
	}
}

fn calculate_projection(canvas_width: u32, canvas_height: u32, screen_width: u32, screen_height: u32) -> Mat4
{
	let canvas_aspect_ratio = canvas_width as f32 / canvas_height as f32;
	let screen_aspect_ratio = screen_width as f32 / screen_height as f32;

	// adjusted canvas dimensions
	let mut adj_canvas_w = canvas_width;
	let mut adj_canvas_h = canvas_height;

	// if the screen is wider than the canvas, make the canvas wider.
	// otherwise, make the canvas taller.
	if screen_aspect_ratio > canvas_aspect_ratio {
		adj_canvas_w = canvas_height * screen_width / screen_height;
	} else {
		adj_canvas_h = canvas_width * screen_height / screen_width;
	}

	let half_width = adj_canvas_w as f32 / 2.0;
	let half_height = adj_canvas_h as f32 / 2.0;
	Mat4::orthographic_lh(-half_width, half_width, -half_height, half_height, 0.0, 1.0)
}

