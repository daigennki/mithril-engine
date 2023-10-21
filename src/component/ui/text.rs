/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use serde::Deserialize;
use shipyard::WorkloadSystem;

use crate::component::{EntityComponent, WantsSystemAdded};

/// UI component meant to render the text to a quad.
#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct UIText
{
	pub text_str: String,
	pub size: f32,
}
impl WantsSystemAdded for UIText
{
	fn add_system(&self) -> Option<(std::any::TypeId, WorkloadSystem)>
	{
		None
	}
}

