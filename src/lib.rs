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
	pub fn new(pref_path: String, game_name: &str) -> Result<GameContext, Box<dyn std::error::Error>>
	{
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
					*control_flow = winit::event_loop::ControlFlow::Exit;
				},
				Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
					self.render_context.recreate_swapchain().unwrap_or_else(|e| {
						log_error(e);
						*control_flow = winit::event_loop::ControlFlow::Exit;
					});
				},
				Event::RedrawEventsCleared => {
					self.draw_in_event_loop().unwrap_or_else(|e| {
						log_error(e);
						*control_flow = winit::event_loop::ControlFlow::Exit;
					});
				}
				_ => (),
			}
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

pub fn run_game(org_name: &str, game_name: &str)
{
	// get preferences path
	// (log, config, and save data files will be saved here)
	let pref_path;
	match get_pref_path(org_name, game_name) {
		Ok(p) => pref_path = p,
		Err(e) => {
			print_error_unlogged(&e);
			return
		}
	}
	println!("Using preferences path: {}", &pref_path);

	// open log file
	let log_file;
	match open_log_file(&pref_path) {
		Ok(l) => log_file = l,
		Err(e) => {
			print_error_unlogged(&e);
			return
		}
	}

	// set up logger
	let logger_config = ConfigBuilder::new()
		.set_time_to_local(true)	// use time in time zone local to system
		.set_time_format_str("%FT%T.%f%Z")	// use RFC 3339 format
		.build();
	let term_logger = TermLogger::new(LevelFilter::Info, logger_config.clone(), TerminalMode::Mixed, ColorChoice::Auto);
	let write_logger = WriteLogger::new(LevelFilter::Info, logger_config, log_file);
    match CombinedLogger::init(vec![ term_logger, write_logger ]) {
		Ok(()) => (),
		Err(e) => {
			print_error_unlogged(&format!("CombinedLogger::init failed: {}", e));
			return;
		}
	}

	log::info!("--- Initializing MithrilEngine... ---");

	// construct GameContext
	let mut gctx;
	match GameContext::new(pref_path, game_name) {
		Ok(g) => gctx = g,
		Err(e) => {
			print_init_error(e);
			return
		}
	}

	// run render loop
	gctx.render_loop();
}


fn log_error(e: Box<dyn std::error::Error>)
{
	log::error!("ERROR: {}", e);
	msgbox::create("Engine Error", &e.to_string(), msgbox::common::IconType::Error) 
		.unwrap_or_else(|mbe| log::error!("Failed to create error message box: {}", mbe));
}

fn print_error_unlogged(e: &str) 
{
	println!("{}", e);
	msgbox::create("Engine error", &e.to_string(), msgbox::common::IconType::Error)
		.unwrap_or_else(|mbe| println!("msgbox::create failed: {}", mbe));
}

fn print_init_error(e: Box<dyn std::error::Error>)
{
	log::error!("ERROR: {}", e);

	let msg_str = format!("Initialization error!\n\n{}", e);
	msgbox::create("Engine error", &msg_str, msgbox::common::IconType::Error)
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

fn open_log_file(pref_path: &str) -> Result<std::fs::File, String>
{
	let log_file_path = format!("{}game.log", &pref_path);
	std::fs::File::create(&log_file_path).or_else(|e| {
		Err(format!("Failed to create log file '{0}': {1}", &log_file_path, e))
	})
}
