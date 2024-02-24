//#![windows_subsystem = "windows"]	// Specify the "windows" subsystem on Windows to hide the console window.
mod fps_counter;

fn main()
{
	mithrilengine::run!("daigennki", "Untitled Game Project", "examples/maps/ferris.yaml");
}
