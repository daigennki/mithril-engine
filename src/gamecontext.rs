use std::io::Write;
use std::sync::Arc;
use vulkano_win::VkSurfaceBuild;
use winit::window::{Window, WindowBuilder};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::physical::PhysicalDevice;
use vulkano::format::Format;

pub struct GameContext 
{
	pref_path: String,
	log_file: std::fs::File,
	event_loop: winit::event_loop::EventLoop<()>,
	vk_dev: Arc<vulkano::device::Device>,
	swapchain: Arc<vulkano::swapchain::Swapchain<Window>>,
	swapchain_images: Vec<Arc<vulkano::image::swapchain::SwapchainImage<Window>>>,
	basic_rp: Arc<vulkano::render_pass::RenderPass>
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

		// get physical device
		let vkpd = get_vk_physical_device(&log_file, &vkinst)?;

		// create window
		let window_surface;
		match create_game_window(&event_loop, game_name, &vkinst) {
			Ok(w) => window_surface = w,
			Err(e) => {
				print_init_error(&log_file, &e.to_string());
				return Err(());
			}
		}
		
		// create logical device
		let (vk_dev, mut queues) = create_vk_logical_device(&log_file, vkpd)?;

		// get queue
		let dev_queue;
		match queues.next() {
			Some(q) => dev_queue = q,
			None => {
				print_init_error(&log_file, "No queues available!");
				return Err(());
			}
		}

		// create swapchain
		let (swapchain, swapchain_images) = create_vk_swapchain(&log_file, &vk_dev, &window_surface)?;

		let basic_rp_result = vulkano::single_pass_renderpass!(vk_dev.clone(),
			attachments: {
				first: {
					load: Clear,
					store: Store,
					format: swapchain.format(),
					samples: 1,
				}
			}, 
			pass: {
				color: [first],
				depth_stencil: {}
			}
		);
		let basic_rp;
		match basic_rp_result {
			Ok(r) => basic_rp = r,
			Err(e) => {
				let error_formatted = format!("Error creating render pass: {}", &e.to_string());
				print_init_error(&log_file, &error_formatted);
				return Err(());
			}
		}

		Ok(GameContext { 
			pref_path: pref_path,
			log_file: log_file,
			event_loop: event_loop,
			vk_dev: vk_dev,
			swapchain: swapchain,
			swapchain_images: swapchain_images,
			basic_rp: basic_rp
		})
	}

	pub fn render_loop(&mut self)
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
}

fn create_vulkan_instance() -> Result<Arc<vulkano::instance::Instance>, String>
{
	let mut app_info = vulkano::app_info_from_cargo_toml!();
	app_info.engine_name = Some(std::borrow::Cow::from("MithrilEngine"));

	let vk_ext = vulkano_win::required_extensions();
	
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

fn get_vk_physical_device<'a>(log_file: &std::fs::File, vkinst: &'a Arc<vulkano::instance::Instance>) 
	-> Result<PhysicalDevice<'a>, ()>
{
	// Get physical device.
	log_info(&log_file, "Available Vulkan physical devices:");
	for pd in PhysicalDevice::enumerate(&vkinst) {
		log_info(&log_file, &pd.properties().device_name);
	}
	
	// Look for a discrete GPU.
	let dgpu = PhysicalDevice::enumerate(&vkinst)
		.find(|pd| pd.properties().device_type == PhysicalDeviceType::DiscreteGpu);
	let vkpd;
	match dgpu {
		Some(g) => vkpd = g,
		None => {
			// If there is no discrete GPU, try to look for an integrated GPU instead.
			let igpu = PhysicalDevice::enumerate(&vkinst)
				.find(|pd| pd.properties().device_type == PhysicalDeviceType::IntegratedGpu);
			match igpu {
				Some(g) => vkpd = g,
				None => {
					// If there are still no GPUs, return with an error.
					print_init_error(&log_file, "No GPUs were found!");
					return Err(());
				}
			}
		}
	}
	// TODO: Check to make sure that the GPU is even capable of the features we need from it.
	
	let string_formatted = format!("Using physical device: {}", &vkpd.properties().device_name);
	log_info(&log_file, &string_formatted);

	return Ok(vkpd);
}

fn create_vk_logical_device(log_file: &std::fs::File, physical_device: PhysicalDevice) 
	-> Result<(Arc<vulkano::device::Device>, vulkano::device::QueuesIter), ()>
{
	// get queue family that supports graphics
	let q_fam;
	match physical_device.queue_families().find(|q| q.supports_graphics()) {
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
	}.union(physical_device.required_extensions());

	let device_tuple;
	match vulkano::device::Device::new(physical_device, &dev_features, &dev_extensions, [(q_fam, 0.5)].iter().cloned()) {
		Ok(d) => device_tuple = d,
		Err(e) => {
			let error_formatted = format!("Failed to create Vulkan logical device: {}", e.to_string());
			print_init_error(&log_file, &error_formatted);
			return Err(());
		}
	}
	let (vk_dev, queues) = device_tuple;

	return Ok((vk_dev, queues));
}

fn create_vk_swapchain(
	log_file: &std::fs::File, 
	device: &Arc<vulkano::device::Device>, 
	surf: &Arc<vulkano::swapchain::Surface<Window>>
) 
	-> Result<(Arc<vulkano::swapchain::Swapchain<Window>>, Vec<Arc<vulkano::image::swapchain::SwapchainImage<Window>>>), ()>
{
	// query surface capabilities
	let surf_caps;
	match surf.capabilities(device.physical_device()) {
		Ok(c) => surf_caps = c,
		Err(e) => {
			let error_formatted = format!("Failed to query surface capabilities: {}", e.to_string());
			print_init_error(&log_file, &error_formatted);
			return Err(());
		}
	}

	let swapchain_result = vulkano::swapchain::Swapchain::start(device.clone(), surf.clone())
		.num_images(surf_caps.min_image_count)
		.format(Format::B8G8R8A8_SRGB)
		.usage(vulkano::image::ImageUsage::color_attachment())
		.build();
	match swapchain_result {
		Ok(s) => return Ok(s),
		Err(e) => {
			let error_formatted = format!("Failed to create swapchain: {}", &e.to_string());
			print_init_error(&log_file, &error_formatted);
			return Err(());
		}
	}
}

fn log_info(mut log_file: &std::fs::File, s: &str) 
{
	println!("{}", s);
	let str_with_newline = format!("{}\n", s);
	match log_file.write_all(str_with_newline.as_bytes()) {
		Ok(()) => (),
		Err(e) => println!("log_info failed to print to log file: {}", &e.to_string())
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
