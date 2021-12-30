/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021, daigennki (@daigennki)

	Licensed under the BSD 3-Clause License.
----------------------------------------------------------------------------- */
#![windows_subsystem = "windows"]
mod gamecontext;
mod util;

fn main() -> Result<(), ()>
{
	// construct GameContext
	let gctx = gamecontext::GameContext::new("daigennki", "MithrilEngine")?;

	// run render loop
	gctx.render_loop();

	return Ok(());
}
