/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod util;
mod rendercontext;

use std::rc::Rc;
use util::log_info;
use winit::event::{ Event, WindowEvent };
use winit::platform::run_return::EventLoopExtRunReturn;

struct GameContext
{
	pref_path: String,
	log_file: Rc<std::fs::File>,
	event_loop: Option<winit::event_loop::EventLoop<()>>,
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
			event_loop: Some(event_loop),
			render_context: render_context
		})
	}

	fn log_error(&self, s: String)
	{
		let log_str = format!("ERROR: {}", s);
		self.print_log(&log_str);
		match msgbox::create("Engine Error", &s, msgbox::common::IconType::Error) {
			Ok(r) => r,
			Err(mbe) => {
				let msgbox_error_str = format!("Failed to create error message box: {}", &mbe.to_string());
				self.print_log(&msgbox_error_str);
			}
		}
	}

	pub fn print_log(&self, s: &str) 
	{
		log_info(&self.log_file, s);
	}

	pub fn render_loop(&mut self)
	{
		let mut event_loop;
		match self.event_loop.take() {
			Some(el) => event_loop = el,
			None => {
				self.log_error(
					"GameContext::event_loop was empty! Did render_loop accidentally get run twice or more?".to_string()
				);
				return;
			}
		}

		event_loop.run_return(move |event, _, control_flow| {
			match event {
				Event::WindowEvent{ event: WindowEvent::CloseRequested, .. } => {
					*control_flow = winit::event_loop::ControlFlow::Exit;
				},
				Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
					//recreate_swapchain = true;
				},
				Event::RedrawEventsCleared => {
					match self.draw_in_event_loop() {
						Ok(()) => (),
						Err(e) => {
							self.log_error(e.to_string());
							*control_flow = winit::event_loop::ControlFlow::Exit;
						}
					}
				}
				_ => (),
			}
		});
	}

	fn draw_in_event_loop(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		let cb = self.render_context.start_commands();

		// placeholder delay
		std::thread::sleep(std::time::Duration::from_millis(16));

		// TODO: finish commands and build command buffer here

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
		Ok(l) => log_file = Rc::new(l),
		Err(e) => {
			print_error_unlogged(&e);
			return
		}
	}

	// construct GameContext
	let mut gctx;
	match GameContext::new(pref_path, log_file.clone(), org_name, game_name) {
		Ok(g) => gctx = g,
		Err(e) => {
			print_init_error(log_file.as_ref(), &e);
			return
		}
	}

	// run render loop
	gctx.render_loop();
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
			_ => Err(format!("Failed to create preferences path: {}", &e.to_string()))
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
	match std::fs::File::create(&log_file_path) {
		Ok(f) => Ok(f),
		Err(e) => Err(format!("Failed to create log file '{0}': {1}", &log_file_path, e))
	}
}
