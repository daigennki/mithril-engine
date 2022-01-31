/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::io::Read;
use std::sync::Arc;
use vulkano_win::VkSurfaceBuild;
use winit::window::{Window, WindowBuilder};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::physical::PhysicalDevice;
use vulkano::format::Format;
use vulkano::pipeline::graphics;
use vulkano::swapchain::Swapchain;
use vulkano::image::swapchain::SwapchainImage;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::render_pass::Subpass;
use vulkano::render_pass::Framebuffer;
use vulkano::sync::{self, FlushError, GpuFuture};
use super::util::log_info;

pub struct RenderContext 
{
	vk_dev: Arc<vulkano::device::Device>,
	window_surface: Arc<vulkano::swapchain::Surface<Window>>,
	swapchain: Arc<vulkano::swapchain::Swapchain<Window>>,
	swapchain_images: Vec<Arc<vulkano::image::swapchain::SwapchainImage<Window>>>,
	framebuffers: Vec<Arc<vulkano::render_pass::Framebuffer>>,
	basic_rp: Arc<vulkano::render_pass::RenderPass>,
	basic_pipeline: Arc<vulkano::pipeline::GraphicsPipeline>,
	q_fam_id: u32,
	dev_queue: Arc<vulkano::device::Queue>,
	cur_cb: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, StandardCommandPoolBuilder>>,
	cur_image_num: usize,
	acquire_future: Option<vulkano::swapchain::SwapchainAcquireFuture<Window>>,
	previous_frame_end: Option<Box<dyn vulkano::sync::GpuFuture>>,
	//rotation_start: std::time::Instant,
	pub need_new_swapchain: bool
}
impl RenderContext
{
	pub fn new(log_file: &std::fs::File, game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) 
		-> Result<RenderContext, Box<dyn std::error::Error>>
	{
		// create Vulkan instance
		let vkinst = create_vulkan_instance()?;

		// create window
		let window_surface = create_game_window(&event_loop, game_name, vkinst.clone()).or_else(|e| Err(e.to_string()))?;

		// create logical device
		let (vk_dev, mut queues) = create_vk_logical_device(&log_file, vkinst.clone())?;

		// get queue family that supports graphics
		let q_fam_id = vk_dev.physical_device().queue_families().find(|q| q.supports_graphics())
			.ok_or("No appropriate queue family found!")?
			.id();

		// get queue
		let dev_queue = queues.next().ok_or("No queues are available!")?;

		// create swapchain
		let (swapchain, swapchain_images) = create_vk_swapchain(vk_dev.clone(), window_surface.clone())?;

		// create basic renderpass
		let basic_rp = vulkano::single_pass_renderpass!(vk_dev.clone(),
			attachments: {
				color: {
					load: Clear,
					store: Store,
					format: swapchain.format(),
					samples: 1,
				}
			}, 
			pass: {
				color: [color],
				depth_stencil: {}
			}
		)?;

		let (framebuffers, basic_pipeline) = setup_pipeline(
			vk_dev.clone(), basic_rp.clone(), swapchain.clone(), &swapchain_images
		)?;

		let previous_frame_end = Some(sync::now(vk_dev.clone()).boxed());
    	//let rotation_start = std::time::Instant::now();
			
		Ok(RenderContext{
			vk_dev: vk_dev,
			window_surface: window_surface,
			swapchain: swapchain,
			swapchain_images: swapchain_images,
			framebuffers: framebuffers,
			basic_rp: basic_rp,
			basic_pipeline: basic_pipeline,
			q_fam_id: q_fam_id,
			dev_queue: dev_queue,
			cur_cb: None,
			cur_image_num: 0,
			acquire_future: None,
			previous_frame_end: previous_frame_end,
			//rotation_start: rotation_start,
			need_new_swapchain: false
		})
	}

	pub fn start_main_commands(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		if self.cur_cb.is_some() {
			return Err(Box::new(CommandBufferAlreadyBuilding))
		}

		if self.need_new_swapchain {
			self.recreate_swapchain()?;
		}

		self.previous_frame_end.as_mut().ok_or("previous_frame_end is `None`!")?.cleanup_finished();

		// TODO: recreate the swapchain in this function when it's suboptimal
		let (image_num, suboptimal, acquire_future) =
			match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
				Ok(r) => r,
				Err(vulkano::swapchain::AcquireError::OutOfDate) => {
					// recreate the swapchain then try again
					self.need_new_swapchain = true;
					return self.start_main_commands();
				}
				Err(e) => return Err(Box::new(e))
			};

		if suboptimal {
			self.need_new_swapchain = true;
		}

		self.cur_image_num = image_num;
		self.acquire_future = Some(acquire_future);

		let q_fam = get_queue_family_from_id(&self.vk_dev, self.q_fam_id)?;
		self.cur_cb = Some(AutoCommandBufferBuilder::primary(self.vk_dev.clone(), q_fam, CommandBufferUsage::OneTimeSubmit)?);

		Ok(())
	}

	pub fn begin_main_render_pass(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		self.cur_cb.as_mut().ok_or(CommandBufferNotBuilding)?
			.begin_render_pass(
				self.framebuffers[self.cur_image_num].clone(),
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
		let built_cb = self.cur_cb.take().ok_or(CommandBufferNotBuilding)?.build()?;
		let acquire_future = self.acquire_future.take().ok_or(CommandBufferNotBuilding)?;

		let future = self.previous_frame_end.take().ok_or(CommandBufferNotBuilding)?
			.join(acquire_future)
			.then_execute(self.dev_queue.clone(), built_cb)?
			.then_swapchain_present(self.dev_queue.clone(), self.swapchain.clone(), self.cur_image_num)
			.then_signal_fence_and_flush();

		match future {
			Ok(future) => {
				self.previous_frame_end = Some(future.boxed());
			}
			Err(FlushError::OutOfDate) => {
				self.recreate_swapchain()?;
				self.previous_frame_end = Some(sync::now(self.vk_dev.clone()).boxed());
			}
			Err(e) => {
				self.previous_frame_end = Some(sync::now(self.vk_dev.clone()).boxed());
				return Err(Box::new(e))
			}
		}
		
		Ok(())
	}

	fn recreate_swapchain(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		let dimensions: [u32; 2] = self.window_surface.window().inner_size().into();

		let (new_swapchain, new_images) = self.swapchain.recreate().dimensions(dimensions).build()?;
		self.swapchain = new_swapchain;
		self.swapchain_images = new_images;

		let (new_framebuffers, new_pipeline) = setup_pipeline(
			self.vk_dev.clone(), self.basic_rp.clone(), self.swapchain.clone(), &self.swapchain_images
		)?;
		self.basic_pipeline = new_pipeline;
		self.framebuffers = new_framebuffers;

		self.need_new_swapchain = false;

		Ok(())
	}
}

fn get_queue_family_from_id(vk_dev: &Arc<vulkano::device::Device>, q_fam_id: u32) 
	-> Result<vulkano::device::physical::QueueFamily, InvalidQueueIDError>
{
	vk_dev.physical_device().queue_family_by_id(q_fam_id).ok_or(InvalidQueueIDError)
}

fn setup_pipeline(
	vk_dev: Arc<vulkano::device::Device>, 
	basic_rp: Arc<vulkano::render_pass::RenderPass>, 
	swapchain: Arc<Swapchain<Window>>,
	swapchain_images: &Vec<Arc<SwapchainImage<Window>>>
) -> Result<(Vec::<Arc<Framebuffer>>, Arc<vulkano::pipeline::GraphicsPipeline>), Box<dyn std::error::Error>>
{
	let basic_rp_subpass = Subpass::from(basic_rp.clone(), 0).ok_or("Subpass for render pass doesn't exist!")?;

	// create frame buffers
	let mut framebuffers = Vec::<Arc<Framebuffer>>::with_capacity(swapchain_images.len());
	for img in swapchain_images.iter() {
		let view = vulkano::image::view::ImageView::new(img.clone())?;
		// TODO: add depth buffers
		framebuffers.push(Framebuffer::start(basic_rp.clone()).add(view)?.build()?);
	}

	// load vertex shader
	let vs = load_spirv(vk_dev.clone(), "shaders/fill_viewport.vert.spv")?;
	let vs_entry = vs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
	
	// load fragment shader
	let fs = load_spirv(vk_dev.clone(), "shaders/ui.frag.spv")?;
	let fs_entry = fs.entry_point("main").ok_or("No valid 'main' entry point in SPIR-V module!")?;
	
	// create pipeline
	let viewport = graphics::viewport::Viewport{ 
		origin: [ 0.0, 0.0 ],
		dimensions: [ swapchain.dimensions()[0] as f32, swapchain.dimensions()[1] as f32 ],
		depth_range: (-1.0..1.0)
	};
	let pipeline = vulkano::pipeline::GraphicsPipeline::start()
		.vertex_shader(vs_entry, ())
		.viewport_state(graphics::viewport::ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
		.fragment_shader(fs_entry, ())
		.render_pass(basic_rp_subpass)
		.build(vk_dev)?;

	Ok((framebuffers, pipeline))
}

fn create_game_window(event_loop: &winit::event_loop::EventLoop<()>, title: &str, vkinst: Arc<vulkano::instance::Instance>) 
	-> Result<Arc<vulkano::swapchain::Surface<Window>>, vulkano_win::CreationError>
{
	WindowBuilder::new()
		.with_inner_size(winit::dpi::PhysicalSize{ width: 1280, height: 720 })
		.with_title(title)
		.with_resizable(false)
		.build_vk_surface(event_loop, vkinst.clone())
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

fn create_vk_logical_device(log_file: &std::fs::File, vkinst: Arc<vulkano::instance::Instance>) 
	-> Result<(Arc<vulkano::device::Device>, vulkano::device::QueuesIter), Box<dyn std::error::Error>>
{
	// Get physical device.
	log_info(&log_file, "Available Vulkan physical devices:");

	for pd in PhysicalDevice::enumerate(&vkinst) {
		let pd_type_str;
		match pd.properties().device_type {
			PhysicalDeviceType::IntegratedGpu => pd_type_str = "Integrated GPU",
			PhysicalDeviceType::DiscreteGpu => pd_type_str = "Discrete GPU",
			PhysicalDeviceType::VirtualGpu => pd_type_str = "Virtual GPU",
			PhysicalDeviceType::Cpu => pd_type_str = "CPU",
			PhysicalDeviceType::Other => pd_type_str = "Other",
		}
		let pd_info = format!("- {} ({})", pd.properties().device_name, pd_type_str);
		log_info(&log_file, &pd_info);
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

	let pd_info_print = format!("Using physical device: {}", physical_device.properties().device_name);
	log_info(&log_file, &pd_info_print);

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

fn create_vk_swapchain(device: Arc<vulkano::device::Device>, surf: Arc<vulkano::swapchain::Surface<Window>>) 
	-> Result<(Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>), Box<dyn std::error::Error>>
{
	// query surface capabilities
	let surf_caps = surf.capabilities(device.physical_device())?;

	Ok(Swapchain::start(device.clone(), surf.clone())
		.num_images(surf_caps.min_image_count)
		.format(Format::B8G8R8A8_SRGB)
		.usage(vulkano::image::ImageUsage::color_attachment())
		.build()?)
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

#[derive(Debug)]
struct InvalidQueueIDError;
impl std::error::Error for InvalidQueueIDError {}
impl std::fmt::Display for InvalidQueueIDError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "The given queue ID was invalid!")
    }
}
