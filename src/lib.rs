/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod render;
pub mod component;
pub mod vertex;

use winit::event::{ Event, WindowEvent };
use simplelog::*;
use glam::*;
use component::ui;
use component::ui::{ canvas::Canvas };
use component::camera::Camera;
use shipyard::{ World, View, ViewMut, Get, UniqueViewMut };
use shipyard::iter::{ IntoIter, IntoWithId };

struct GameContext
{
	//pref_path: String,
	render_context: render::RenderContext,
	world: World
}
impl GameContext
{
	// game context "constructor"
	pub fn new(org_name: &str, game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) 
		-> Result<GameContext, Box<dyn std::error::Error>>
	{
		/*let pref_path =*/ setup_log(org_name, game_name)?;

		log::info!("--- Initializing MithrilEngine... ---");

		// get command line arguments
		// let args: Vec<String> = std::env::args().collect();

		let mut render_ctx = render::RenderContext::new(game_name, &event_loop)?;

		let mut world = World::new();

		// add some 3D entities for testing
		world.add_unique(Camera::new(&mut render_ctx, [ 1.0, 3.0, 3.0 ].into(), [ 0.0, 0.0, 0.0 ].into())?)?;

		world.add_entity(component::new_triangle(&mut render_ctx, [ 0.0, 0.0, 0.0 ].into(), Vec3::ONE, [ 0.1, 0.0, 0.0, 0.8 ].into())?);
		world.add_entity(component::new_triangle(&mut render_ctx, [ 0.2, 0.0, 0.2 ].into(), Vec3::ONE, [ 0.0, 0.1, 0.0, 0.8 ].into())?);
		world.add_entity(component::new_triangle(&mut render_ctx, [ 0.4, 0.0, 0.4 ].into(), Vec3::ONE, [ 0.0, 0.0, 0.1, 0.8 ].into())?);
		world.add_entity(component::new_triangle(&mut render_ctx, [ 0.6, 0.0, 0.6 ].into(), Vec3::ONE, [ 0.1, 0.1, 0.0, 0.8 ].into())?);


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
		self.render_context.bind_3d_pipeline();
		self.world.run(|camera: UniqueViewMut<Camera>|
		{
			camera.bind(&mut self.render_context)
		})?;
		self.world.run_with_data(draw_3d, &mut self.render_context)??;

		// Draw the UI element components.
		//self.render_context.bind_ui_pipeline();
		//self.world.run_with_data(draw_ui_elements, &mut self.render_context)??;
		
		self.render_context.end_render_pass()?;

		self.render_context.submit_commands()?;

		Ok(())
	}
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
		transform.bind_descriptor_set(render_ctx);

		// draw 3D meshes
		match meshes.get(eid) {
			Ok(c) => {
				c.draw(render_ctx)?
			},
			Err(_) => ()
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
		match meshes.get(eid) {
			Ok(c) => c.draw(render_ctx)?,
			Err(_) => ()
		}
		match texts.get(eid) {
			Ok(c) => c.draw(render_ctx)?,
			Err(_) => ()
		}
	}

	Ok(())
}

pub fn run_game(org_name: &str, game_name: &str)
{
	let event_loop = winit::event_loop::EventLoop::new();

	match GameContext::new(org_name, game_name, &event_loop) {
		Ok(mut gctx) => event_loop.run(move |event, _, control_flow| {
			match event {
				Event::WindowEvent{ event: WindowEvent::CloseRequested, .. } => {
					*control_flow = winit::event_loop::ControlFlow::Exit;	// TODO: show exit confirmation dialog here
				},
				_ => (),
			};
			
			gctx.handle_event(&event).unwrap_or_else(|e| {
				log_error(e);
				*control_flow = winit::event_loop::ControlFlow::Exit;
			});
		}),	
		Err(e) => log_error(e)
	}
}

// Get preferences path, set up logging, and return the preferences path.
fn setup_log(org_name: &str, game_name: &str) -> Result<String, Box<dyn std::error::Error>>
{
	let pref_path = get_pref_path(org_name, game_name)?;	// log, config, and save data files will be saved here
	println!("Using preferences path: {}", &pref_path);

	// open log file
	let log_file_path = format!("{}game.log", &pref_path);
	let log_file = std::fs::File::create(&log_file_path).or_else(|e| {
		Err(format!("Failed to create log file '{}': {}", &log_file_path, e))
	})?;

	// set up logger
	let logger_config = ConfigBuilder::new()
		.set_time_offset_to_local().unwrap_or_else(|e| {
			println!("WARNING: simplelog::ConfigBuilder::set_time_offset_to_local failed!");
			e
		})	// use time in time zone local to system
		.set_time_format_rfc3339()	// use RFC 3339 format
		.build();
	let term_log_level;
	#[cfg(debug_assertions)] 
	{
		term_log_level = LevelFilter::Debug;
	}
	#[cfg(not(debug_assertions))] 
	{
		term_log_level = LevelFilter::Info;
	}	
	let term_logger = TermLogger::new(term_log_level, logger_config.clone(), TerminalMode::Mixed, ColorChoice::Auto);
	let write_logger = WriteLogger::new(term_log_level /*LevelFilter::Info*/, logger_config, log_file);
    CombinedLogger::init(vec![ term_logger, write_logger ])?;

	Ok(pref_path)
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

fn create_pref_path(prefix: &str, org_name: &str, game_name: &str) -> Result<String, String>
{
	let pref_path = format!("{}/{}/{}/", prefix, org_name, game_name);

	// try to create the path if it doesn't exist
	match std::fs::create_dir_all(&pref_path) {
		Ok(()) => Ok(pref_path),
		Err(e) => match e.kind() {
			std::io::ErrorKind::AlreadyExists => {
				println!("Preferences path already exists, skipping creation...");
				Ok(pref_path)
			},
			_ => Err(format!("Failed to create preferences path: {}", e))
		}
	}
}
fn get_pref_path(org_name: &str, game_name: &str) -> Result<String, String>
{
	#[cfg(target_family = "windows")]
	let path_prefix = std::env::var("APPDATA");
	#[cfg(target_family = "unix")]
	let path_prefix = std::env::var("XDG_DATA_HOME");
	
	match path_prefix {
		Ok(env_result) => Ok(create_pref_path(&env_result, org_name, game_name)?),
		Err(e) => {
			#[cfg(target_family = "windows")]
			return Err(format!("Failed to get preferences path: {}", e));
			#[cfg(target_family = "unix")]
			{
				println!("XDG_DATA_HOME was invalid ({}), trying HOME instead...", e);
				match std::env::var("HOME") {
					Ok(env_result) => {
						let pref_prefix = format!("{}/.local/share", env_result);
						Ok(create_pref_path(&pref_prefix, org_name, game_name)?)
					},
					Err(e) => Err(format!("Failed to get preferences path: {}", e))
				}
			}
		}
	}
}
