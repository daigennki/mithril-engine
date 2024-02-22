/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
pub mod mesh;
pub mod text;

use glam::*;
use serde::Deserialize;

use crate::component::{ComponentSystems, EntityComponent};
use crate::SystemBundle;

#[derive(Default, shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct UITransform
{
	pub position: IVec2,
	pub scale: Option<Vec2>, // leave as `None` to use scale from image of another component

	                         // TODO: parent-child relationship
}
impl ComponentSystems for UITransform {}
