//#![windows_subsystem = "windows"]	// Specify the "windows" subsystem on Windows to hide the console window.
mod fps_counter;

fn main()
{
	mithril_engine::run!("daigennki", "Untitled Game Project", "examples/maps/ferris.yaml");
}
