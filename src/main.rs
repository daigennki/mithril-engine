/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
#![windows_subsystem = "windows"]
mod gamecontext;

fn main() -> Result<(), ()>
{
	gamecontext::run_game("daigennki", "MithrilEngine")
}
