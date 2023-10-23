//#![windows_subsystem = "windows"]	// Specify the "windows" subsystem on Windows to hide the console window.

// Here are the modules for the custom components.
mod camera_controller;
mod fps_counter;

fn main()
{
	// Run the game with this developer name, game name, and first map.
	mithrilengine::run_game("daigennki", "Small island", "examples/small_island/maps/small_island.yaml");
}
