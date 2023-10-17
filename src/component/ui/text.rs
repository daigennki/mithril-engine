/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

/// UI component meant to render the text to a quad.
#[derive(shipyard::Component)]
#[track(All)]
pub struct Text
{
	pub text_str: String,
	pub size: f32,
}

