//#![windows_subsystem = "windows"]	// Specify the "windows" subsystem on Windows to hide the console window.

// Here are the modules for the custom components.
mod camera_controller;
mod fps_counter;

fn main()
{
	// Run the game with this developer name, game name, and first map.
	mithril_engine::run!("daigennki", "Small island", "examples/maps/small_island.yaml");
}
