use crate::util::log_info;

use std::sync::Arc;
use vulkano_win::VkSurfaceBuild;
use winit::window::{Window, WindowBuilder};

pub struct GameContext 
{
	pref_path: String,
	log_file: std::fs::File,
	event_loop: winit::event_loop::EventLoop<()>,
	//game_window: Window,
	window_surface: Arc<vulkano::swapchain::Surface<Window>>,
	vkinst: Arc<vulkano::instance::Instance>,
	vk_dev: Arc<vulkano::device::Device>
}
impl GameContext 
{
	// game context "constructor"
	pub fn new(org_name: &str, game_name: &str) -> Result<GameContext, ()> 
	{
		// get preferences path
		// (log, config, and save data files will be saved here)
		let pref_path = get_pref_path(org_name, game_name)?;
		println!("Using preferences path: {}", &pref_path);

		// open log file
		let log_file = open_log_file(&pref_path)?;
		
		// print start date and time
		let dt_str = format!("INIT {}", chrono::Local::now().to_rfc3339());
		log_info(&log_file, &dt_str);

		// get command line arguments
		// let args: Vec<String> = std::env::args().collect();

		// create event loop
		let event_loop = winit::event_loop::EventLoop::new();

		// create Vulkan instance
		let vkinst;
		match create_vulkan_instance() {
			Ok(vki) => vkinst = vki,
			Err(e) => {
				print_init_error(&log_file, &e.to_string());
				return Err(());
			}
		}

		// create window
		let window_surface;
		match create_game_window(&event_loop, game_name, &vkinst) {
			Ok(w) => window_surface = w,
			Err(e) => {
				print_init_error(&log_file, &e.to_string());
				return Err(());
			}
		}

		// get physical device
		// TODO: check physical device type (prioritize discrete graphics)
		let vkpd: vulkano::device::physical::PhysicalDevice;
		match vulkano::device::physical::PhysicalDevice::from_index(&vkinst, 0) {
			Some(pd) => vkpd = pd,
			None => {
				print_init_error(&log_file, "No physical devices found!");
				return Err(());
			}
		}
		let string_formatted = format!("Using physical device: {}", &vkpd.properties().device_name);
		log_info(&log_file, &string_formatted);

		// get queue family that supports graphics
		let q_fam;
		match vkpd.queue_families().find(|q| q.supports_graphics()) {
			Some(q) => q_fam = q,
			None => {
				print_init_error(&log_file, "No appropriate queue family found!");
				return Err(());
			}
		}
		
		// create logical device
		let dev_features = vulkano::device::Features{
			image_cube_array: true,
			independent_blend: true,
			sampler_anisotropy: true,
			texture_compression_bc: true,
			geometry_shader: true,
			..vulkano::device::Features::none()
		};
		let dev_extensions = vulkano::device::DeviceExtensions{
			khr_swapchain: true,
			..vulkano::device::DeviceExtensions::none()
		}.union(vkpd.required_extensions());

		let device_tuple;
		match vulkano::device::Device::new(vkpd, &dev_features, &dev_extensions, [(q_fam, 0.5)].iter().cloned()) {
			Ok(d) => device_tuple = d,
			Err(e) => {
				let error_formatted = format!("Failed to create Vulkan logical device: {}", e.to_string());
				print_init_error(&log_file, &error_formatted);
				return Err(());
			}
		}
		let (vk_dev, mut queues) = device_tuple;

		// get queue
		let dev_queue;
		match queues.next() {
			Some(q) => dev_queue = q,
			None => {
				print_init_error(&log_file, "No queues available!");
				return Err(());
			}
		}

		// query surface capabilities
		let surf_caps;
		match window_surface.capabilities(vkpd) {
			Ok(c) => surf_caps = c,
			Err(e) => {
				let error_formatted = format!("Failed to query surface capabilities: {}", e.to_string());
				print_init_error(&log_file, &error_formatted);
				return Err(());
			}
		}

		let composite_alpha = surf_caps.supported_composite_alpha.iter().next().unwrap();
		let format = surf_caps.supported_formats[0].0;
		let dimensions: [u32; 2] = window_surface.window().inner_size().into();

		Ok(GameContext { 
			pref_path: pref_path,
			log_file: log_file,
			event_loop: event_loop,
			window_surface: window_surface,
			vkinst: vkinst,
			vk_dev: vk_dev
		})
	}

	pub fn render_loop(&self)
	{
		match self.render_loop_inner() {
			Ok(()) => (),
			Err(e) => self.render_loop_error(e)
		}
		self.print_log("Success.");
	}

	pub fn print_log(&self, s: &str) 
	{
		log_info(&self.log_file, s);
	}

	fn render_loop_error(&self, e: Box<dyn std::error::Error>) 
	{
		self.print_log(&format!("ERROR: {}", &e.to_string()));
		match msgbox::create("Engine Error", &e.to_string(), msgbox::common::IconType::Error) {
			Ok(r) => r,
			Err(mbe) => {
				let msgbox_error_str = format!("Failed to create error message box: {}", &mbe.to_string());
				self.print_log(&msgbox_error_str);
			}
		}
	}

	fn render_loop_inner(&self) -> Result<(), Box<dyn std::error::Error>> 
	{
		// wait for 2 seconds
		std::thread::sleep(std::time::Duration::from_millis(2000));

		Ok(())
	}
}

fn create_game_window(event_loop: &winit::event_loop::EventLoop<()>, title: &str, vkinst: &Arc<vulkano::instance::Instance>) 
	-> Result<Arc<vulkano::swapchain::Surface<Window>>, vulkano_win::CreationError>
{
	return WindowBuilder::new()
		.with_inner_size(winit::dpi::PhysicalSize{ width: 1280, height: 720 })
		.with_title(title)
		.build_vk_surface(event_loop, vkinst.clone());
		//.build(event_loop);
}

fn create_vulkan_instance() -> Result<std::sync::Arc<vulkano::instance::Instance>, String>
{
	let mut app_info = vulkano::app_info_from_cargo_toml!();
	app_info.engine_name = Some(std::borrow::Cow::from("MithrilEngine"));

	let vk_ext = vulkano::instance::InstanceExtensions::none();
	
	let vk_layer_list: Vec<_>;
	match vulkano::instance::layers_list() {
		Ok(layers_list) => {
			vk_layer_list = layers_list.filter(|l| l.description().contains("VK_LAYER_KHRONOS_validation")).collect();
		},
		Err(e) => {
			return Err(e.to_string());
		}
	}
	let vk_layer_names = vk_layer_list.iter().map(|l| l.name());

	match vulkano::instance::Instance::new(Some(&app_info), vulkano::Version::V1_2, &vk_ext, vk_layer_names) {
		Ok(vki) => Ok(vki),
		Err(e) => Err(e.to_string())
	}
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

fn get_pref_path(org_name: &str, game_name: &str) -> Result<String, ()>
{
	/*match sdl2::filesystem::pref_path(org_name, game_name) {
		Ok(s) => return Ok(s),
		Err(e) => {
			let error_formatted = format!("Failed to get preferences path: {}", &e.to_string());
			print_error_unlogged(&error_formatted);
			return Err(());
		}
	}*/
	// TODO: try to create the path if it doesn't exist
	// TODO: get preferences path for other platforms
	let path_prefix = std::env::var("APPDATA");
	match path_prefix {
		Ok(env_result) => {
			let pref_path = format!("{}/{}/{}", env_result, org_name, game_name);
			return Ok(pref_path);
		}
		Err(e) => {
			let error_formatted = format!("Failed to get preferences path: {}", &e.to_string());
			print_error_unlogged(&error_formatted);
			return Err(());
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
