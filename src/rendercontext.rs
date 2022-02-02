/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod swapchain;
mod pipeline;


use std::sync::Arc;
use vulkano_win::VkSurfaceBuild;
use winit::window::WindowBuilder;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::physical::PhysicalDevice;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;

pub struct RenderContext 
{
	vk_dev: Arc<vulkano::device::Device>,
	swapchain: swapchain::Swapchain,
	basic_pipeline: pipeline::Pipeline,
	q_fam_id: u32,
	dev_queue: Arc<vulkano::device::Queue>,
	cur_cb: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, StandardCommandPoolBuilder>>
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) 
		-> Result<RenderContext, Box<dyn std::error::Error>>
	{
		// create Vulkan instance
		let vkinst = create_vulkan_instance(game_name)?;

		// create logical device
		let (vk_dev, mut queues) = create_vk_logical_device(vkinst.clone())?;

		// get queue family that supports graphics
		let q_fam_id = vk_dev.physical_device().queue_families().find(|q| q.supports_graphics())
			.ok_or("No appropriate queue family found!")?
			.id();

		// get queue
		let dev_queue = queues.next().ok_or("No queues are available!")?;

		// create window
		let window_surface = WindowBuilder::new()
			.with_inner_size(winit::dpi::PhysicalSize{ width: 1280, height: 720 })
			.with_title(game_name)
			.with_resizable(false)
			.build_vk_surface(&event_loop, vkinst)?;

		// create swapchain
		let swapchain = swapchain::Swapchain::new(vk_dev.clone(), window_surface)?;

		// create pipeline
		let dimensions = swapchain.dimensions();
		let basic_pipeline = pipeline::Pipeline::new(
			vk_dev.clone(), 
			"shaders/fill_viewport.vert.spv",
			Some("shaders/ui.frag.spv"),
			swapchain.render_pass(), 
			dimensions[0], dimensions[1]
		)?;
			
		Ok(RenderContext{
			vk_dev: vk_dev,
			swapchain: swapchain,
			basic_pipeline: basic_pipeline,
			q_fam_id: q_fam_id,
			dev_queue: dev_queue,
			cur_cb: None
		})
	}

	pub fn start_main_commands(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		if self.cur_cb.is_some() {
			return Err(Box::new(CommandBufferAlreadyBuilding))
		}

		let q_fam = self.vk_dev.physical_device().queue_family_by_id(self.q_fam_id)
			.ok_or("The given queue ID was invalid!")?;

		self.cur_cb = Some(
			AutoCommandBufferBuilder::primary(self.vk_dev.clone(), q_fam, CommandBufferUsage::OneTimeSubmit)?
		);

		Ok(())
	}

	pub fn begin_main_render_pass(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		self.cur_cb.as_mut().ok_or(CommandBufferNotBuilding)?
			.begin_render_pass(
				self.swapchain.get_next_image()?,
				vulkano::command_buffer::SubpassContents::Inline,
				vec![[0.0, 0.0, 1.0, 1.0].into()/*, 1f32.into()*/],
			)?;
		Ok(())
	}

	pub fn end_main_render_pass(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		self.cur_cb.as_mut().ok_or(CommandBufferNotBuilding)?.end_render_pass()?;
		Ok(())
	}

	pub fn submit_commands(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		self.swapchain.submit_commands(self.cur_cb.take().ok_or(CommandBufferNotBuilding)?, self.dev_queue.clone())
	}

	pub fn recreate_swapchain(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		self.swapchain.recreate_swapchain()?;
		let dimensions = self.swapchain.dimensions();
		self.basic_pipeline = pipeline::Pipeline::new(
			self.vk_dev.clone(), 
			"shaders/fill_viewport.vert.spv",
			Some("shaders/ui.frag.spv"),
			self.swapchain.render_pass(), 
			dimensions[0], dimensions[1]
		)?;

		Ok(())
	}
}

fn create_vulkan_instance(game_name: &str) -> Result<Arc<vulkano::instance::Instance>, Box<dyn std::error::Error>>
{
	let mut app_info = vulkano::app_info_from_cargo_toml!();
	app_info.application_name = Some(std::borrow::Cow::from(game_name));
	app_info.engine_name = Some(std::borrow::Cow::from("MithrilEngine"));
	app_info.engine_version = app_info.application_version.clone();

	let vk_ext = vulkano_win::required_extensions();
	#[cfg(debug_assertions)]
	let vk_layer_list: Vec<_>;
	let vk_layer_names;
	
	// only use the validation layer in debug builds
	#[cfg(debug_assertions)]
	{
		vk_layer_list = vulkano::instance::layers_list()?
			.filter(|l| l.description().contains("VK_LAYER_KHRONOS_validation"))
			.collect();
		vk_layer_names = vk_layer_list.iter().map(|l| l.name()).collect::<Vec<&str>>();
	}
	#[cfg(not(debug_assertions))]
	{
		vk_layer_names = None;
	}
	
	Ok(vulkano::instance::Instance::new(Some(&app_info), vulkano::Version::V1_2, &vk_ext, vk_layer_names)?)
}

fn create_vk_logical_device(vkinst: Arc<vulkano::instance::Instance>) 
	-> Result<(Arc<vulkano::device::Device>, vulkano::device::QueuesIter), Box<dyn std::error::Error>>
{
	// Get physical device.
	log::info!("Available Vulkan physical devices:");

	for pd in PhysicalDevice::enumerate(&vkinst) {
		let pd_type_str = match pd.properties().device_type {
			PhysicalDeviceType::IntegratedGpu => "Integrated GPU",
			PhysicalDeviceType::DiscreteGpu => "Discrete GPU",
			PhysicalDeviceType::VirtualGpu => "Virtual GPU",
			PhysicalDeviceType::Cpu => "CPU",
			PhysicalDeviceType::Other => "Other",
		};
		log::info!("- {} ({})", pd.properties().device_name, pd_type_str);
	}
	
	// Look for a discrete GPU.
	let dgpu = PhysicalDevice::enumerate(&vkinst)
		.find(|pd| pd.properties().device_type == PhysicalDeviceType::DiscreteGpu);
	let physical_device;
	match dgpu {
		Some(g) => physical_device = g,
		None => {
			// If there is no discrete GPU, try to look for an integrated GPU instead.
			physical_device = PhysicalDevice::enumerate(&vkinst)
				.find(|pd| pd.properties().device_type == PhysicalDeviceType::IntegratedGpu)
				.ok_or("No GPUs were found!")?;
		}
	}

	log::info!("Using physical device: {}", physical_device.properties().device_name);

	// get queue family that supports graphics
	let q_fam = physical_device.queue_families().find(|q| q.supports_graphics())
		.ok_or("No appropriate queue family found!")?;

	// select features and extensions.
	// the ones chosen here are practically universally supported by any device with Vulkan support.
	let dev_features = vulkano::device::Features{
		image_cube_array: true,
		independent_blend: true,
		sampler_anisotropy: true,
		texture_compression_bc: true,	// we might need to change this to ASTC or ETC2 if we want to support mobile platforms
		geometry_shader: true,
		..vulkano::device::Features::none()
	};
	let dev_extensions = vulkano::device::DeviceExtensions{
		khr_swapchain: true,
		..vulkano::device::DeviceExtensions::none()
	}.union(physical_device.required_extensions());

	// create logical device
	let use_queue_families = [(q_fam, 0.5)];
	Ok(vulkano::device::Device::new(physical_device, &dev_features, &dev_extensions, use_queue_families)?)
}

#[derive(Debug)]
struct CommandBufferNotBuilding;
impl std::error::Error for CommandBufferNotBuilding {}
impl std::fmt::Display for CommandBufferNotBuilding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "The RenderContext's current command buffer is not building! Did you forget to call `start_main_commands`?")
    }
}

#[derive(Debug)]
struct CommandBufferAlreadyBuilding;
impl std::error::Error for CommandBufferAlreadyBuilding {}
impl std::fmt::Display for CommandBufferAlreadyBuilding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "The RenderContext's current command buffer is already building! Did you forget to submit the previous one?")
    }
}
