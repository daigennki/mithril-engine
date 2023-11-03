//#![windows_subsystem = "windows"]	// Specify the "windows" subsystem on Windows to hide the console window.
mod fps_counter;

fn main()
{
	mithrilengine::run_game("daigennki", "Untitled Game Project", "examples/ferris/maps/ferris.yaml");
}
