/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod util;
mod rendercontext;

use std::rc::Rc;
use util::log_info;

pub struct GameContext
{
	pref_path: String,
	log_file: Rc<std::fs::File>,
	event_loop: winit::event_loop::EventLoop<()>,
	render_context: rendercontext::RenderContext
}
impl GameContext
{
	// game context "constructor"
	pub fn new(pref_path: String, log_file: Rc<std::fs::File>, org_name: &str, game_name: &str) -> Result<GameContext, String>
	{
		// print start date and time
		let dt_str = format!("INIT {}", chrono::Local::now().to_rfc3339());
		log_info(&log_file, &dt_str);

		// get command line arguments
		// let args: Vec<String> = std::env::args().collect();

		// create event loop
		let event_loop = winit::event_loop::EventLoop::new();

		let render_context = rendercontext::RenderContext::new(log_file.as_ref(), game_name, &event_loop)?;

		Ok(GameContext { 
			pref_path: pref_path,
			log_file: log_file,
			event_loop: event_loop,
			render_context: render_context
		})
	}

	pub fn print_log(&self, s: &str) 
	{
		log_info(&self.log_file, s);
	}

	pub fn render_loop(&mut self)
	{
		match self.render_loop_inner() {
			Ok(()) => (),
			Err(e) => {
				let log_str = format!("ERROR: {}", e);
				self.print_log(&log_str);
				match msgbox::create("Engine Error", &e.to_string(), msgbox::common::IconType::Error) {
					Ok(r) => r,
					Err(mbe) => {
						let msgbox_error_str = format!("Failed to create error message box: {}", &mbe.to_string());
						self.print_log(&msgbox_error_str);
					}
				}
			}
		}
	}

	fn render_loop_inner(&mut self) -> Result<(), Box<dyn std::error::Error>> 
	{
		let cb = self.render_context.start_commands()?;

		// wait for 2 seconds
		std::thread::sleep(std::time::Duration::from_millis(2000));

		// TODO: finish commands and build command buffer here

		Ok(())
	}
}

pub fn run_game(org_name: &str, game_name: &str) -> Result<(), ()> 
{
	// get preferences path
	// (log, config, and save data files will be saved here)
	let pref_path = get_pref_path(org_name, game_name)?;
	println!("Using preferences path: {}", &pref_path);

	// open log file
	let log_file = Rc::new(open_log_file(&pref_path)?);
	
	// construct GameContext
	let mut gctx;
	match GameContext::new(pref_path, log_file.clone(), org_name, game_name) {
		Ok(g) => gctx = g,
		Err(e) => {
			print_init_error(log_file.as_ref(), &e);
			return Err(())
		}
	}

	// run render loop
	gctx.render_loop();

	Ok(())
}



fn print_error_unlogged(s: &str) 
{
	println!("{}", &s);
	match msgbox::create("Engine error", &s, msgbox::common::IconType::Error) {
		Ok(r) => r,
		Err(mbe) => println!("msgbox::create failed: {}", &mbe.to_string())
	}
}

fn print_init_error(log_file: &std::fs::File, e: &str)
{
	let error_formatted = format!("ERROR: {}", e);
	log_info(log_file, &error_formatted);

	let msg_str = format!("Initialization error!\n\n{}", e);
	match msgbox::create("Engine error", &msg_str, msgbox::common::IconType::Error) {
		Ok(r) => r,
		Err(mbe) => {
			let mbe_str = format!("Failed to create error message box: {}", &mbe.to_string());
			log_info(log_file, &mbe_str);
		}
	}
}

fn create_pref_path(prefix: &str, org_name: &str, game_name: &str) -> Result<String, ()>
{
	let pref_path = format!("{}/{}/{}/", prefix, org_name, game_name);

	// try to create the path if it doesn't exist
	match std::fs::create_dir_all(&pref_path) {
		Ok(()) => return Ok(pref_path),
		Err(e) => match e.kind() {
			std::io::ErrorKind::AlreadyExists => {
				println!("Preferences path already exists, skipping creation...");
				return Ok(pref_path);
			},
			_ => {
				let error_formatted = format!("Failed to create preferences path: {}", &e.to_string());
				print_error_unlogged(&error_formatted);
				return Err(());
			}
		}
	}
}
fn get_pref_path(org_name: &str, game_name: &str) -> Result<String, ()>
{
	#[cfg(target_family = "windows")]
	let path_prefix = std::env::var("APPDATA");
	#[cfg(target_family = "unix")]
	let path_prefix = std::env::var("XDG_DATA_HOME");
	
	match path_prefix {
		Ok(env_result) => {
			return Ok(create_pref_path(&env_result, org_name, game_name)?);
		},
		Err(e) => {
			#[cfg(target_family = "windows")]
			{
				let error_formatted = format!("Failed to get preferences path: {}", &e.to_string());
				print_error_unlogged(&error_formatted);
				return Err(());
			}
			#[cfg(target_family = "unix")]
			{
				println!("XDG_DATA_HOME was invalid ({}), trying HOME instead...", &e.to_string());
				match std::env::var("HOME") {
					Ok(env_result) => {
						let pref_prefix = format!("{}/.local/share", env_result);
						return Ok(create_pref_path(&pref_prefix, org_name, game_name)?);
					},
					Err(e) => {
						let error_formatted = format!("Failed to get preferences path: {}", &e.to_string());
						print_error_unlogged(&error_formatted);
						return Err(());
					}
				}
			}
		}
	}
}

fn open_log_file(pref_path: &str) -> Result<std::fs::File, ()>
{
	let log_file_path = format!("{}game.log", &pref_path);
	match std::fs::File::create(&log_file_path) {
		Ok(f) => return Ok(f),
		Err(e) => {
			let error_formatted = format!("Failed to create log file '{0}': {1}", &log_file_path, &e.to_string());
			print_error_unlogged(&error_formatted);
			return Err(());
		}
	}
}
