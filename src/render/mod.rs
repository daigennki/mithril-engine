/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod swapchain;
pub mod pipeline;
pub mod texture;

use std::sync::Arc;
use vulkano_win::VkSurfaceBuild;
use winit::window::WindowBuilder;
use vulkano::device::physical::{ PhysicalDeviceType, PhysicalDevice, QueueFamily };
use vulkano::device::{ DeviceCreationError, Queue };
use vulkano::command_buffer::{ AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, DrawError };
use vulkano::command_buffer::{ SubpassContents };
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::graphics::vertex_input::VertexBuffersCollection;
use vulkano::pipeline::graphics::input_assembly::{ PrimitiveTopology, Index };
use vulkano::descriptor_set::{ DescriptorSetsCollection, WriteDescriptorSet, PersistentDescriptorSet };
use vulkano::sampler::Sampler;
use vulkano::format::{ Format };
use vulkano::buffer::{ ImmutableBuffer, BufferUsage, TypedBufferAccess };
use vulkano::sync::{ GpuFuture };
use vulkano::image::{ ImageDimensions, MipmapsCount };

pub struct RenderContext 
{
	vk_dev: Arc<vulkano::device::Device>,
	swapchain: swapchain::Swapchain,
	dev_queue: Arc<vulkano::device::Queue>,
	cur_cb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,

	upload_futures: Option<Box<dyn vulkano::sync::GpuFuture>>,
	upload_futures_count: usize,

	// TODO: figure out a better way to allow user-defined shader pipelines
	ui_pipeline: pipeline::Pipeline,
	world_pipeline: pipeline::Pipeline	// 3D world pipeline
}
impl RenderContext
{
	pub fn new(game_name: &str, event_loop: &winit::event_loop::EventLoop<()>) 
		-> Result<RenderContext, Box<dyn std::error::Error>>
	{
		let vkinst = create_vulkan_instance(game_name)?;
		let (physical_device, q_fam) = get_physical_device(&vkinst)?;
		let (vk_dev, mut queues) = create_vk_logical_device(physical_device, [(q_fam, 0.5)])?;
		let dev_queue = queues.next().ok_or("No queues are available!")?;

		// create window
		let window_surface = WindowBuilder::new()
			.with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
			.with_title(game_name)
			.with_resizable(false)
			.build_vk_surface(&event_loop, vk_dev.instance().clone())?;

		// create swapchain
		let swapchain = swapchain::Swapchain::new(vk_dev.clone(), window_surface)?;
		let dim = swapchain.dimensions();
		
		// create UI pipeline
		let ui_pipeline = pipeline::Pipeline::new_from_yaml("ui.yaml", swapchain.render_pass(), dim[0], dim[1])?;

		// create 3D pipeline	
		let world_pipeline = pipeline::Pipeline::new_from_yaml("world.yaml", swapchain.render_pass(), dim[0], dim[1])?;

		let cur_cb = AutoCommandBufferBuilder::primary(vk_dev.clone(), q_fam, CommandBufferUsage::OneTimeSubmit)?;
			
		Ok(RenderContext{
			vk_dev: vk_dev,
			swapchain: swapchain,
			dev_queue: dev_queue,
			cur_cb: cur_cb,
			upload_futures: None,
			ui_pipeline: ui_pipeline,
			world_pipeline: world_pipeline,
			upload_futures_count: 0
		})
	}

	pub fn begin_main_render_pass(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		let (next_img_fb, resize_viewports) = self.swapchain.get_next_image()?;

		if resize_viewports {
			let new_dimensions = self.swapchain.dimensions();
			log::debug!("Recreating pipelines with new viewport...");
			// recreate pipelines with new viewport
			for pl in [ &mut self.ui_pipeline ] {
				pl.resize_viewport(new_dimensions[0], new_dimensions[1])?;
			}
		}

		self.cur_cb.begin_render_pass(next_img_fb, SubpassContents::Inline, [[0.1, 0.1, 0.1, 1.0].into()])?;
		Ok(())
	}

	/*pub fn begin_gui_render_pass(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		self.cur_cb.begin_render_pass(
			self.swapchain.get_current_image(),
			vulkano::command_buffer::SubpassContents::SecondaryCommandBuffers,
			vec![[0.0, 0.0, 1.0, 1.0].into()/*, 1f32.into()*/],
		)?;
		Ok(())
	}*/

	pub fn end_render_pass(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		self.cur_cb.end_render_pass()?;
		Ok(())
	}

	pub fn submit_commands(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		let q_fam = self.vk_dev.active_queue_families().next()
			.ok_or("There are no active queue families in the logical device!")?;

		// Leave a new command buffer builder in place of the one we're about to take to build and submit.
		let mut swap_cb = AutoCommandBufferBuilder::primary(self.vk_dev.clone(), q_fam, CommandBufferUsage::OneTimeSubmit)?;
		std::mem::swap(&mut swap_cb, &mut self.cur_cb);

		let submit_futures = self.upload_futures.take();	// consume the futures to join them upon submission
		if submit_futures.is_some() {
			log::debug!("Joining a future of {} futures.", self.upload_futures_count);
		}
		self.upload_futures_count = 0;
		self.swapchain.submit_commands(swap_cb.build()?, self.dev_queue.clone(), submit_futures)
	}

	pub fn new_texture(&mut self, path: &std::path::Path) -> Result<texture::Texture, Box<dyn std::error::Error>>
	{
		let (tex, upload_future) = texture::Texture::new(self.dev_queue.clone(), path)?;

		self.upload_futures = Some(match self.upload_futures.take() {
			Some(f) => upload_future.join(f).boxed(),
			None => upload_future.boxed()
		});
		self.upload_futures_count += 1;

		Ok(tex)
	}

	pub fn new_texture_from_iter<Px, I>(&mut self,	
		iter: I, 
		vk_fmt: Format, 
		dimensions: ImageDimensions,
		mip: MipmapsCount
	) 
	-> Result<texture::Texture, Box<dyn std::error::Error>>
	where
		[Px]: vulkano::buffer::BufferContents,
		I: IntoIterator<Item = Px>,
		I::IntoIter: ExactSizeIterator
	{
		let (tex, upload_future) = texture::Texture::new_from_iter(
			self.dev_queue.clone(), 
			iter,
			vk_fmt, 
			dimensions, 
			mip
		)?;

		self.upload_futures = Some(match self.upload_futures.take() {
			Some(f) => upload_future.join(f).boxed(),
			None => upload_future.boxed()
		});
		self.upload_futures_count += 1;

		Ok(tex)
	}

	pub fn new_buffer<D,T>(&mut self, data: D, usage: BufferUsage) 
		-> Result<Arc<ImmutableBuffer<[T]>>, Box<dyn std::error::Error>>
		where
			D: IntoIterator<Item = T>,
			D::IntoIter: ExactSizeIterator,
			[T]: vulkano::buffer::BufferContents, 
	{
		let (buf, upload_future) = ImmutableBuffer::from_iter(data, usage, self.dev_queue.clone())?;
		
		self.upload_futures = Some(match self.upload_futures.take() {
			Some(f) => upload_future.join(f).boxed(),
			None => upload_future.boxed()
		});
		self.upload_futures_count += 1;

		Ok(buf)
	} 
	
	/*pub fn bind_pipeline(&mut self, pipeline: &pipeline::Pipeline) -> Result<(), CommandBufferNotBuilding>
	{
		pipeline.bind(self.cur_cb.as_mut().ok_or(CommandBufferNotBuilding)?);
		Ok(())
	}*/

	pub fn bind_ui_pipeline(&mut self)
	{
		self.ui_pipeline.bind(&mut self.cur_cb);
	}

	pub fn new_ui_descriptor_set(&self, set: usize, writes: impl IntoIterator<Item = WriteDescriptorSet>)
		-> Result<Arc<PersistentDescriptorSet>, Box<dyn std::error::Error>>
	{
		self.ui_pipeline.new_descriptor_set(set, writes)
	}

	pub fn bind_ui_descriptor_set<S>(&mut self, first_set: u32, descriptor_sets: S) 
		where S: DescriptorSetsCollection
	{
		self.cur_cb.bind_descriptor_sets(PipelineBindPoint::Graphics, self.ui_pipeline.layout(), first_set, descriptor_sets);
	}

	pub fn bind_3d_pipeline(&mut self)
	{
		self.world_pipeline.bind(&mut self.cur_cb);
	}

	pub fn new_3d_descriptor_set(&self, set: usize, writes: impl IntoIterator<Item = WriteDescriptorSet>)
		-> Result<Arc<PersistentDescriptorSet>, Box<dyn std::error::Error>>
	{
		self.world_pipeline.new_descriptor_set(set, writes)
	}

	pub fn bind_3d_descriptor_set<S>(&mut self, first_set: u32, descriptor_sets: S)
		where S: DescriptorSetsCollection
	{
		self.cur_cb.bind_descriptor_sets(PipelineBindPoint::Graphics, self.world_pipeline.layout(), first_set, descriptor_sets);
	}


	pub fn bind_vertex_buffers<V>(&mut self, first_binding: u32, vertex_buffers: V)
		where V: VertexBuffersCollection
	{
		self.cur_cb.bind_vertex_buffers(first_binding, vertex_buffers);
	}

	pub fn bind_index_buffers<Ib, I>(&mut self, index_buffer: Arc<Ib>)
		where 
			Ib: TypedBufferAccess<Content = [I]> + 'static,
			I: Index + 'static
	{
		self.cur_cb.bind_index_buffer(index_buffer);
	}

	pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32)
		-> Result<(), DrawError>
	{
		self.cur_cb.draw(vertex_count, instance_count, first_vertex, first_instance)?;
		Ok(())
	}

	/*pub fn surface(&self) -> Arc<vulkano::swapchain::Surface<winit::window::Window>>
	{
		self.swapchain.surface()
	}
	pub fn queue(&self) -> Arc<vulkano::device::Queue>
	{
		self.dev_queue.clone()
	}

	pub fn submit_secondary(&mut self, s: vulkano::command_buffer::SecondaryAutoCommandBuffer) -> Result<(), ExecuteCommandsError>
	{
		self.cur_cb.execute_commands(s)?;
		Ok(())
	}

	pub fn swapchain_dimensions(&self) -> [u32; 2]
	{
		self.swapchain.dimensions()
	}

	pub fn get_main_subpass(&self) -> vulkano::render_pass::Subpass
	{
		self.swapchain.render_pass().first_subpass()
	}*/

	/*pub fn wait_for_fence(&self) -> Result<(), FlushError>
	{
		self.swapchain.wait_for_fence()
	}*/
}

fn create_vulkan_instance(game_name: &str) -> Result<Arc<vulkano::instance::Instance>, Box<dyn std::error::Error>>
{
	let vk_ext = vulkano_win::required_extensions();
	
	// only use the validation layer in debug builds
	// TODO: re-enable this later, since vulkano is a bit borked as of v0.29.0 and doesn't get along well with the validation
	// layer; see https://github.com/vulkano-rs/vulkano/issues/1858
	/*#[cfg(debug_assertions)]
	let vk_layers = vec!["VK_LAYER_KHRONOS_validation".to_string()];
	#[cfg(not(debug_assertions))]*/
	let vk_layers: Vec<String> = vec![];

	let mut inst_create_info = vulkano::instance::InstanceCreateInfo::application_from_cargo_toml();
	inst_create_info.application_name = Some(game_name.to_string());
	inst_create_info.engine_name = Some("MithrilEngine".to_string());
	inst_create_info.engine_version = inst_create_info.application_version.clone();
	inst_create_info.enabled_extensions = vk_ext;
	inst_create_info.enabled_layers = vk_layers;
	inst_create_info.max_api_version = Some(vulkano::Version::V1_2);
	
	Ok(vulkano::instance::Instance::new(inst_create_info)?)
}

fn decode_driver_version(version: u32, vendor_id: u32) -> (u32, u32, u32, u32)
{
	// NVIDIA
	if vendor_id == 4318 {
		return ((version >> 22) & 0x3ff,
		(version >> 14) & 0x0ff,
		(version >> 6) & 0x0ff,
		version & 0x003f)
	}
	
	// Intel (Windows only)
	#[cfg(target_family = "windows")]
	if vendor_id == 0x8086 {
		return ((version >> 14),
		version & 0x3fff,
		0, 0)
	}

	// others (use Vulkan version convention)
	((version >> 22),
	(version >> 12) & 0x3ff,
	version & 0xfff, 0)
}
fn print_physical_devices<'a>(vkinst: &'a Arc<vulkano::instance::Instance>)
{
	log::info!("Available Vulkan physical devices:");
	for pd in PhysicalDevice::enumerate(&vkinst) {
		let pd_type_str = match pd.properties().device_type {
			PhysicalDeviceType::IntegratedGpu => "Integrated GPU",
			PhysicalDeviceType::DiscreteGpu => "Discrete GPU",
			PhysicalDeviceType::VirtualGpu => "Virtual GPU",
			PhysicalDeviceType::Cpu => "CPU",
			PhysicalDeviceType::Other => "Other",
		};
		let driver_ver = decode_driver_version(pd.properties().driver_version, pd.properties().vendor_id);
		let api_ver = pd.properties().api_version;
		
		log::info!("{}: {} ({}), driver '{}' version {}.{}.{}.{} (Vulkan {}.{}.{})", 
			pd.index(), 
			pd.properties().device_name, 
			pd_type_str,
			pd.properties().driver_name.clone().unwrap_or("unknown driver".to_string()), 
			driver_ver.0, driver_ver.1, driver_ver.2, driver_ver.3,
			api_ver.major, api_ver.minor, api_ver.patch
		);
	}
}
fn print_queue_families<'a>(queue_families: impl ExactSizeIterator<Item = QueueFamily<'a>>)
{
	log::info!("Available physical device queue families:");
	for qf in queue_families {
		let qf_type = if qf.supports_graphics() {
			"graphics"
		} else if qf.supports_compute() {
			"compute"
		} else {
			"unknown"
		};
		let explicit_transfer = if qf.explicitly_supports_transfers() {
			"explicit"
		} else {
			"implicit"
		};
		log::info!("{}: {}, {} transfer support, {} queue(s)", qf.id(), qf_type, explicit_transfer, qf.queues_count());
	}
}

/// Get the most appropriate GPU, along with a graphics queue family.
fn get_physical_device<'a>(vkinst: &'a Arc<vulkano::instance::Instance>) 
	-> Result<(PhysicalDevice<'a>, QueueFamily<'a>), Box<dyn std::error::Error>>
{	
	print_physical_devices(vkinst);

	// Look for a discrete GPU.
	let dgpu = PhysicalDevice::enumerate(&vkinst)
		.find(|pd| pd.properties().device_type == PhysicalDeviceType::DiscreteGpu);
	let physical_device = match dgpu {
		Some(g) => g,

		// If there is no discrete GPU, try to look for an integrated GPU instead.
		None => PhysicalDevice::enumerate(&vkinst)
			.find(|pd| pd.properties().device_type == PhysicalDeviceType::IntegratedGpu)
			.ok_or("No GPUs were found!")?
	};
	log::info!("Using physical device: {}", physical_device.properties().device_name);

	// get queue family that supports graphics
	print_queue_families(physical_device.queue_families());
	let q_fam = physical_device.queue_families().find(|q| q.supports_graphics())
			.ok_or("No appropriate queue family found!")?;

	Ok((physical_device, q_fam))
}

fn create_vk_logical_device<'a, I>(physical_device: PhysicalDevice, queue_families: I) 
	-> Result<(Arc<vulkano::device::Device>, impl ExactSizeIterator<Item = Arc<Queue>>), DeviceCreationError>
	where I: IntoIterator<Item = (QueueFamily<'a>, f32)>
{
	// Select features and extensions.
	// The ones chosen here are practically universally supported by any device with Vulkan support.
	let dev_features = vulkano::device::Features{
		image_cube_array: true,
		independent_blend: true,
		sampler_anisotropy: true,
		texture_compression_bc: true,	// change this to ASTC or ETC2 if we want to support mobile platforms
		geometry_shader: true,
		..vulkano::device::Features::none()
	};
	let dev_extensions = vulkano::device::DeviceExtensions{
		khr_swapchain: true,
		..vulkano::device::DeviceExtensions::none()
	}.union(physical_device.required_extensions());

	let mut queue_create_infos: Vec<vulkano::device::QueueCreateInfo<'a>> = Vec::new();
	for qf in queue_families {
		queue_create_infos.push(vulkano::device::QueueCreateInfo::family(qf.0));
	}

	let dev_create_info = vulkano::device::DeviceCreateInfo{
		enabled_extensions: dev_extensions,
		enabled_features: dev_features,
		queue_create_infos: queue_create_infos,
		..Default::default()
	};

	vulkano::device::Device::new(physical_device, dev_create_info)
}
