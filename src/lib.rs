/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod util;
mod rendercontext;

use winit::event::{ Event, WindowEvent };
use winit::platform::run_return::EventLoopExtRunReturn;
use simplelog::*;

struct GameContext
{
	pref_path: String,
	event_loop: Option<winit::event_loop::EventLoop<()>>,
	render_context: rendercontext::RenderContext
}
impl GameContext
{
	// game context "constructor"
	pub fn new(org_name: &str, game_name: &str) -> Result<GameContext, Box<dyn std::error::Error>>
	{
		let pref_path = log_setup(org_name, game_name)?;

		log::info!("--- Initializing MithrilEngine... ---");

		// get command line arguments
		// let args: Vec<String> = std::env::args().collect();

		// create event loop
		let event_loop = winit::event_loop::EventLoop::new();

		let render_context = rendercontext::RenderContext::new(game_name, &event_loop)?;

		Ok(GameContext { 
			pref_path: pref_path,
			event_loop: Some(event_loop),
			render_context: render_context
		})
	}

	pub fn render_loop(&mut self)
	{
		let mut event_loop;
		match self.event_loop.take() {
			Some(el) => event_loop = el,
			None => {
				log_error("GameContext::event_loop was empty! Did render_loop accidentally get run twice or more?".into());
				return;
			}
		}

		event_loop.run_return(move |event, _, control_flow| {
			match event {
				Event::WindowEvent{ event: WindowEvent::CloseRequested, .. } => {
					*control_flow = winit::event_loop::ControlFlow::Exit;	// TODO: exit confirmation dialog here
					Ok(())
				},
				Event::WindowEvent { event: WindowEvent::Resized(_), .. } => self.render_context.recreate_swapchain(),
				Event::RedrawEventsCleared => self.draw_in_event_loop(),
				_ => Ok(()),
			}.unwrap_or_else(|e|{
				log_error(e);
				*control_flow = winit::event_loop::ControlFlow::Exit;
			});
		});
	}

	fn draw_in_event_loop(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		self.render_context.start_main_commands()?;
		self.render_context.begin_main_render_pass()?;

		// draw stuff here

		self.render_context.end_main_render_pass()?;
		self.render_context.submit_commands()?;

		Ok(())
	}
}

// Get preferences path, set up logging, and return the preferences path.
fn log_setup(org_name: &str, game_name: &str) -> Result<String, Box<dyn std::error::Error>>
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
		.set_time_to_local(true)	// use time in time zone local to system
		.set_time_format_str("%FT%T.%f%Z")	// use RFC 3339 format
		.build();
	let term_logger = TermLogger::new(LevelFilter::Info, logger_config.clone(), TerminalMode::Mixed, ColorChoice::Auto);
	let write_logger = WriteLogger::new(LevelFilter::Info, logger_config, log_file);
    CombinedLogger::init(vec![ term_logger, write_logger ])?;

	Ok(pref_path)
}

pub fn run_game(org_name: &str, game_name: &str)
{
	// construct GameContext
	match GameContext::new(org_name, game_name) {
		Ok(mut g) => g.render_loop(),	// run render loop
		Err(e) => log_error(e)	// exit with error
	}
}

fn log_error(e: Box<dyn std::error::Error>)
{
	if log::log_enabled!(log::Level::Error) {
		log::error!("ERROR: {}", e);
	} else {
		println!("ERROR: {}", e);
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
