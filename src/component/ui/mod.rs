/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod canvas;
pub mod mesh;
pub mod text;

use glam::*;
use serde::Deserialize;
use shipyard::WorkloadSystem;

use crate::component::{EntityComponent, WantsSystemAdded};

#[derive(Default, shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct UITransform
{
	pub position: IVec2,
	pub scale: Option<Vec2>,	// leave as `None` to use scale from image of another component

	// TODO: parent-child relationship
}
impl WantsSystemAdded for UITransform
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		None
	}
}

