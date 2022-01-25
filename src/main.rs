/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)

	Licensed under the BSD 3-Clause License.
----------------------------------------------------------------------------- */
#![windows_subsystem = "windows"]
mod gamecontext;

fn main() -> Result<(), ()>
{
	gamecontext::run_game("daigennki", "MithrilEngine")
}
