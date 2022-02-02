/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod swapchain;

use std::io::Read;
use std::sync::Arc;
use vulkano_win::VkSurfaceBuild;
use winit::window::{Window, WindowBuilder};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::physical::PhysicalDevice;
use vulkano::pipeline::graphics;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::render_pass::Subpass;

pub struct RenderContext 
{
	vk_dev: Arc<vulkano::device::Device>,
	swapchain: swapchain::Swapchain,
	basic_pipeline: Arc<vulkano::pipeline::GraphicsPipeline>,
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
		let vkinst = create_vulkan_instance()?;

		// create window
		let window_surface = create_game_window(&event_loop, game_name, vkinst.clone())?;

		// create logical device
		let (vk_dev, mut queues) = create_vk_logical_device(vkinst.clone())?;

		// get queue family that supports graphics
		let q_fam_id = vk_dev.physical_device().queue_families().find(|q| q.supports_graphics())
			.ok_or("No appropriate queue family found!")?
			.id();

		// get queue
		let dev_queue = queues.next().ok_or("No queues are available!")?;

		// create swapchain
		let swapchain = swapchain::Swapchain::new(vk_dev.clone(), window_surface.clone())?;

		let basic_pipeline = setup_pipeline(
			vk_dev.clone(), swapchain.get_render_pass(), 1280.0, 720.0	// TODO: set the proper width and height from the swapchain here
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
		self.basic_pipeline = setup_pipeline(
			self.vk_dev.clone(), self.swapchain.get_render_pass(), 1280.0, 720.0
		)?;

		Ok(())
	}
}

fn setup_pipeline(vk_dev: Arc<vulkano::device::Device>, rp: Arc<vulkano::render_pass::RenderPass>, width: f32, height: f32)
	-> Result<Arc<vulkano::pipeline::GraphicsPipeline>, Box<dyn std::error::Error>>
{
	let rp_subpass = Subpass::from(rp.clone(), 0).ok_or("Subpass for render pass doesn't exist!")?;

	// load vertex shader
	let vs = load_spirv(vk_dev.clone(), "shaders/fill_viewport.vert.spv")?;
	let vs_entry = vs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
	
	// load fragment shader
	let fs = load_spirv(vk_dev.clone(), "shaders/ui.frag.spv")?;
	let fs_entry = fs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
	
	// create pipeline
	let viewport = graphics::viewport::Viewport{ 
		origin: [ 0.0, 0.0 ],
		dimensions: [ width, height ],
		depth_range: (-1.0..1.0)
	};
	let pipeline = vulkano::pipeline::GraphicsPipeline::start()
		.vertex_shader(vs_entry, ())
		.viewport_state(graphics::viewport::ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
		.fragment_shader(fs_entry, ())
		.render_pass(rp_subpass)
		.build(vk_dev)?;

	Ok(pipeline)
}

fn create_game_window(event_loop: &winit::event_loop::EventLoop<()>, title: &str, vkinst: Arc<vulkano::instance::Instance>) 
	-> Result<Arc<vulkano::swapchain::Surface<Window>>, vulkano_win::CreationError>
{
	WindowBuilder::new()
		.with_inner_size(winit::dpi::PhysicalSize{ width: 1280, height: 720 })
		.with_title(title)
		.with_resizable(false)
		.build_vk_surface(event_loop, vkinst)
}

fn create_vulkan_instance() -> Result<Arc<vulkano::instance::Instance>, Box<dyn std::error::Error>>
{
	let mut app_info = vulkano::app_info_from_cargo_toml!();
	app_info.engine_name = Some(std::borrow::Cow::from("MithrilEngine"));

	let vk_ext = vulkano_win::required_extensions();
	let vk_layer_list: Vec<_> = vulkano::instance::layers_list()?
		.filter(|l| l.description().contains("VK_LAYER_KHRONOS_validation"))
		.collect();
	let vk_layer_names = vk_layer_list.iter().map(|l| l.name());

	Ok(vulkano::instance::Instance::new(Some(&app_info), vulkano::Version::V1_2, &vk_ext, vk_layer_names)?)
}

fn create_vk_logical_device(vkinst: Arc<vulkano::instance::Instance>) 
	-> Result<(Arc<vulkano::device::Device>, vulkano::device::QueuesIter), Box<dyn std::error::Error>>
{
	// Get physical device.
	log::info!("Available Vulkan physical devices:");

	for pd in PhysicalDevice::enumerate(&vkinst) {
		let pd_type_str;
		match pd.properties().device_type {
			PhysicalDeviceType::IntegratedGpu => pd_type_str = "Integrated GPU",
			PhysicalDeviceType::DiscreteGpu => pd_type_str = "Discrete GPU",
			PhysicalDeviceType::VirtualGpu => pd_type_str = "Virtual GPU",
			PhysicalDeviceType::Cpu => pd_type_str = "CPU",
			PhysicalDeviceType::Other => pd_type_str = "Other",
		}
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
	// TODO: Check to make sure that the GPU is even capable of the features we need from it.

	log::info!("Using physical device: {}", physical_device.properties().device_name);

	// get queue family that supports graphics
	let q_fam = physical_device.queue_families().find(|q| q.supports_graphics())
		.ok_or("No appropriate queue family found!")?;

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

	let use_queue_families = [(q_fam, 0.5)];
	Ok(vulkano::device::Device::new(physical_device, &dev_features, &dev_extensions, use_queue_families)?)
}

fn load_spirv(device: Arc<vulkano::device::Device>, filename: &str) 
	-> Result<Arc<vulkano::shader::ShaderModule>, Box<dyn std::error::Error>>
{
	let mut spv_file = std::fs::File::open(filename)
		.or_else(|e| Err(format!("Failed to open SPIR-V shader file: {}", e)))?;

	let mut spv_data: Vec<u8> = Vec::new();
	spv_file.read_to_end(&mut spv_data)
		.or_else(|e| Err(format!("Failed to read SPIR-V shader file: {}", e)))?;

	Ok(unsafe { vulkano::shader::ShaderModule::from_bytes(device, &spv_data) }?)
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
