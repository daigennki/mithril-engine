/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use glam::*;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vertex2
{
	pub x: f32,
	pub y: f32
}
impl Vertex2
{
	pub fn new(x: f32, y: f32) -> Vertex2
	{
		Vertex2{ x: x, y: y }
	}
	pub fn new_from_vec2(pos: Vec2) -> Vertex2
	{
		Vertex2{ x: pos.x, y: pos.y }
	}
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Vertex3
{
	pub x: f32,
	pub y: f32,
	pub z: f32
}
impl Vertex3
{
	pub fn new(x: f32, y: f32, z: f32) -> Vertex3
	{
		Vertex3{ x: x, y: y, z: z }
	}
	pub fn new_from_vec2(pos: Vec3) -> Vertex3
	{
		Vertex3{ x: pos.x, y: pos.y, z: pos.z }
	}
}
