/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod canvas;
pub mod mesh;
pub mod text;

use glam::*;
use shipyard::WorkloadSystem;
use std::path::Path;

use crate::component::WantsSystemAdded;

#[derive(Default, shipyard::Component)]
#[track(All)]
pub struct Transform
{
	pub pos: IVec2,
	pub scale: Option<Vec2>,	// leave as `None` to use scale from image of another component

	// TODO: parent-child relationship
}
impl WantsSystemAdded for Transform
{
	fn add_system(&self) -> Option<(std::any::TypeId, WorkloadSystem)>
	{
		None
	}
}

/// Convenience function: create a tuple of `Transform` and `Mesh` to display an image loaded from a file on the UI.
pub fn new_image(path: &Path, pos: IVec2) -> (Transform, mesh::Mesh)
{
	let img_transform = Transform { pos, scale: None };
	let img_mesh = mesh::Mesh {
		image_path: path.to_path_buf(),
		..Default::default()
	};

	(img_transform, img_mesh)
}

/// Convenience function: create a tuple of `Transform` and `Text` to display text.
pub fn new_text(text_str: String, size: f32, pos: IVec2) -> (Transform, text::Text)
{
	let transform = Transform { pos, scale: None };
	let text_component = text::Text { text_str, size };

	(transform, text_component)
}
