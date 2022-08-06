/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod render;
pub mod component;

use std::path::{ Path, PathBuf };
use winit::event::{ Event, WindowEvent };
use simplelog::*;
use serde::Deserialize;
use shipyard::{ World, View, ViewMut, Get, UniqueViewMut };
use shipyard::iter::{ IntoIter, IntoWithId };

use component::ui;
use component::ui::{ canvas::Canvas };
use component::camera::Camera;
use component::DeferGpuResourceLoading;
//use entities::new_triangle;

#[cfg(debug_assertions)]
use LevelFilter::Debug as EngineLogLevel;
#[cfg(not(debug_assertions))]
use LevelFilter::Info as EngineLogLevel;


struct GameContext
{
	//pref_path: String,
	render_context: render::RenderContext,
	world: World
}
impl GameContext
{
	// game context "constructor"
	pub fn new(org_name: &str, game_name: &str, start_map: &str, event_loop: &winit::event_loop::EventLoop<()>) 
		-> Result<GameContext, Box<dyn std::error::Error>>
	{
		/*let pref_path =*/ setup_log(org_name, game_name)?;

		log::info!("--- Initializing MithrilEngine... ---");
		// get command line arguments
		// let args: Vec<String> = std::env::args().collect();

		let mut render_ctx = render::RenderContext::new(game_name, &event_loop)?;

		let mut world = load_map(&mut render_ctx, start_map)?;

		world.add_unique(Camera::new(&mut render_ctx, [ 1.0, 3.0, 3.0 ].into(), [ 0.0, 0.0, 0.0 ].into())?)?;		

		// add some UI entities for testing
		world.add_unique(Canvas::new(1280, 720)?)?;

		world.add_entity(ui::new_image(&mut render_ctx, "test_image.png", [ 0, 0 ].into())?);
		world.add_entity(ui::new_text(&mut render_ctx, "Hello World!", 32.0, [ -200, -200 ].into())?);
		

		// Update the projection matrix on UI `Transform` components.
		// TODO: use tracking instead, when it gets implemented in shipyard stable
		world.run(|mut canvas: UniqueViewMut<Canvas>, mut transforms: ViewMut<ui::Transform>| 
			-> Result<(), Box<dyn std::error::Error>> 
		{
			for (eid, mut transform) in (&mut transforms).iter().with_id() {
				transform.update_projection(&mut render_ctx, canvas.projection())?;
				canvas.add_child(eid);	// TODO: only do this upon component insertion, when tracking is implemented
			}

			Ok(())
		})??;

		let gctx = GameContext { 
			//pref_path: pref_path,
			render_context: render_ctx,
			world: world
		};

		Ok(gctx)
	}

	pub fn handle_event(&mut self, event: &Event<()>) -> Result<(), Box<dyn std::error::Error>>
	{
		match event {
			Event::RedrawEventsCleared => self.draw_in_event_loop(),
			_ => Ok(())
		}
	}

	fn draw_in_event_loop(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		self.render_context.begin_main_render_pass()?;

		// Draw the 3D stuff
		self.render_context.bind_pipeline("World")?;
		self.world.run(|camera: UniqueViewMut<Camera>| camera.bind(&mut self.render_context))??;
		self.world.run_with_data(draw_3d, &mut self.render_context)??;

		// Draw the UI element components.
		self.render_context.bind_pipeline("UI")?;
		self.world.run_with_data(draw_ui_elements, &mut self.render_context)??;
		
		self.render_context.end_render_pass()?;

		self.render_context.submit_commands()?;

		Ok(())
	}
}

#[derive(Deserialize)]
struct WorldData
{
	uniques: Vec<serde_yaml::Value>,
	entities: Vec<Vec<serde_yaml::Value>>
}
impl TryInto<World> for WorldData
{
	type Error = Box<dyn std::error::Error>;

	fn try_into(self) -> Result<World, Self::Error>
	{
		let mut world = World::new();

		for entity in self.entities {
			let eid = world.add_entity(());
			for component in entity {
				match component {
					serde_yaml::Value::Tagged(tagged) =>{
						// TODO: figure out how to (de)serialize prefabs
						// TODO: figure out a way to add user-defined components
						if tagged.tag == "Transform" {
							world.add_component(eid, (serde_yaml::value::from_value::<component::Transform>(tagged.value)?,))
						} else if tagged.tag == "Mesh" {
							world.add_component(eid, (serde_yaml::value::from_value::<component::mesh::Mesh>(tagged.value)?,))
						}
					},
					_ => ()
				}
			}
		}

		Ok(world)
	}
}

fn load_map(render_ctx: &mut render::RenderContext, file: &str) -> Result<World, Box<dyn std::error::Error>>
{
	let yaml_string = String::from_utf8(std::fs::read(Path::new("maps").join(file))?)?;
	let world_data: WorldData = serde_yaml::from_str(&yaml_string)?;
	let world: World = world_data.try_into()?;

	// finish loading GPU resources for components
	// TODO: maybe figure out a way to get trait objects from shipyard
	world.run(|mut components: ViewMut<component::Transform>| -> Result<(), Box<dyn std::error::Error>> {
		for mut component in (&mut components).iter() {
			component.finish_loading(render_ctx)?;
		}
		Ok(())
	})??;
	world.run(|mut components: ViewMut<component::mesh::Mesh>| -> Result<(), Box<dyn std::error::Error>> {
		for mut component in (&mut components).iter() {
			component.finish_loading(render_ctx)?;
		}
		Ok(())
	})??;

	Ok(world)
}

/// Draw 3D objects.
/// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
fn draw_3d(
	render_ctx: &mut render::RenderContext,
	transforms: View<component::Transform>,
	meshes: View<component::mesh::Mesh>
)
	-> Result<(), Box<dyn std::error::Error>>
{
	for (eid, transform) in transforms.iter().with_id() {
		transform.bind_descriptor_set(render_ctx)?;

		// draw 3D meshes
		if let Ok(c) = meshes.get(eid) {
			c.draw(render_ctx)?;
		}
	}

	Ok(())
}


/// Draw UI elements.
/// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
fn draw_ui_elements(
	render_ctx: &mut render::RenderContext, 
	transforms: View<ui::Transform>, 
	meshes: View<ui::mesh::Mesh>,
	texts: View<ui::text::Text>
)
	-> Result<(), Box<dyn std::error::Error>>
{	
	for (eid, transform) in transforms.iter().with_id() {
		transform.bind_descriptor_set(render_ctx)?;

		// draw UI meshes
		// TODO: how do we respect the render order?
		if let Ok(c) = meshes.get(eid) {
			c.draw(render_ctx)?;
		}
		if let Ok(c) = texts.get(eid) {
			c.draw(render_ctx)?;
		}
	}

	Ok(())
}

/// Run the game. This should go in your `main.rs`.
/// `org_name` and `game_name` will be used for the data directory.
/// `game_name` will also be used for the window title.
/// `start_map` is the first map (level/world) to be loaded.
pub fn run_game(org_name: &str, game_name: &str, start_map: &str)
{
	let event_loop = winit::event_loop::EventLoop::new();

	GameContext::new(org_name, game_name, start_map, &event_loop)
		.and_then(|mut gctx| event_loop.run(move |event, _, control_flow| {
			match event {
				Event::WindowEvent{ event: WindowEvent::CloseRequested, .. } => {
					*control_flow = winit::event_loop::ControlFlow::Exit;	// TODO: show exit confirmation dialog here
				},
				_ => (),
			};
			
			if let Err(e) = gctx.handle_event(&event) {
				log_error(e);
				*control_flow = winit::event_loop::ControlFlow::Exit;
			}
		}))
		.unwrap_or_else(|e| log_error(e));
}

// Get data path, set up logging, and return the data path.
fn setup_log(org_name: &str, game_name: &str) -> Result<PathBuf, Box<dyn std::error::Error>>
{
	let data_path = dirs::data_dir().ok_or("Failed to get data directory")?.join(org_name).join(game_name);
	println!("Using data directory: {}", data_path.display());

	// Create the game data directory. Log, config, and save data files will be saved here.
	std::fs::create_dir_all(&data_path).or_else(|e| Err(format!("Failed to create data directory: {}", e)))?;
	
	// open log file
	let log_file_path = data_path.join("game.log");
	let log_file = std::fs::File::create(&log_file_path)
		.or_else(|e| Err(format!("Failed to create '{}': {}", log_file_path.display(), e)))?;

	// set up logger
	let logger_config = ConfigBuilder::new()
		.set_time_offset_to_local()
		.unwrap_or_else(|config_builder| {
			println!("WARNING: simplelog::ConfigBuilder::set_time_offset_to_local failed! Using UTC+0 instead.");
			config_builder
		})	
		.set_time_format_rfc3339()
		.build();

	let term_logger = TermLogger::new(EngineLogLevel, logger_config.clone(), TerminalMode::Mixed, ColorChoice::Auto);
	let write_logger = WriteLogger::new(EngineLogLevel, logger_config, log_file);
    CombinedLogger::init(vec![ term_logger, write_logger ])?;

	Ok(data_path)
}

fn log_error(e: Box<dyn std::error::Error>)
{
	if log::log_enabled!(log::Level::Error) {
		log::error!("{}", e);
	} else {
		println!("{}", e);
	}
	msgbox::create("Engine Error", &e.to_string(), msgbox::common::IconType::Error) 
		.unwrap_or_else(|mbe| log::error!("Failed to create error message box: {}", mbe));
}

