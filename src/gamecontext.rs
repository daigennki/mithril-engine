use crate::util::log_info;

pub struct GameContext 
{
	pref_path: String,
	log_file: std::fs::File,
	sdlc: sdl2::Sdl,
	sdl_vss: sdl2::VideoSubsystem,
	game_window: sdl2::video::Window,
	vkinst: std::sync::Arc<vulkano::instance::Instance>
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

		// initialize SDL2
		let sdl_context;
		match sdl2::init() {
			Ok(sc) => sdl_context = sc,
			Err(e) => {
				print_init_error(&log_file, &e);
				return Err(());
			}
		}
		
		// initialize SDL2 video subsystem
		let sdl_vss;
		match sdl_context.video() {
			Ok(vss) => sdl_vss = vss,
			Err(e) => {
				print_init_error(&log_file, &e);
				return Err(());
			}
		}

		// create window
		let gwnd;
		match create_game_window(&sdl_vss, game_name) {
			Ok(w) => gwnd = w,
			Err(e) => {
				print_init_error(&log_file, &e.to_string());
				return Err(());
			}
		}

		// create Vulkan instance
		let vkinst;
		match create_vulkan_instance() {
			Ok(vki) => vkinst = vki,
			Err(e) => {
				print_init_error(&log_file, &e.to_string());
				return Err(());
			}
		}

		// get physical device
		// TODO: check physical device type (prioritize discrete graphics)
		let mut index = 0;
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

		// get queue families from physical device
		/*let q_fam;
		for queue_family in vkpd.queue_families() {
			if queue_family.supports_graphics() {
				q_fam = queue_family;
				break;
			}
		}
		
		// get logical device
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
		};

		match vulkano::device::Device::new(
			vkpd, 
			&dev_features, 
			&vkpd.required_extensions().union(&dev_extensions), 
			vkpd.queue_families()) {

		}*/

		Ok(GameContext { 
			pref_path: pref_path,
			log_file: log_file,
			sdlc: sdl_context,
			sdl_vss: sdl_vss,
			game_window: gwnd,
			vkinst: vkinst
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

fn create_game_window(vss: &sdl2::VideoSubsystem, title: &str) -> Result<sdl2::video::Window, sdl2::video::WindowBuildError>
{
	let wnd_result = vss.window(title, 1280, 720)
		.position_centered()
		.allow_highdpi()
		.vulkan()
		.build();
	return wnd_result;
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
	match sdl2::filesystem::pref_path(org_name, game_name) {
		Ok(s) => return Ok(s),
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
