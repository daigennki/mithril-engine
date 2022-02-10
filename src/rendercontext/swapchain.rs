/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use winit::window::Window;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::format::Format;
use vulkano::render_pass::{ RenderPass, Framebuffer };
use vulkano::sync::{ FlushError, GpuFuture, FenceSignalFuture};
use vulkano::swapchain::{ SwapchainAcquireFuture, PresentFuture };

pub struct Swapchain
{
	vk_dev: Arc<vulkano::device::Device>,
	swapchain: Arc<vulkano::swapchain::Swapchain<Window>>,
	swapchain_images: Vec<Arc<vulkano::image::swapchain::SwapchainImage<Window>>>,
	swapchain_rp: Arc<RenderPass>,
	framebuffers: Vec<Arc<Framebuffer>>,
	cur_image_num: usize,
	acquire_future: Option<SwapchainAcquireFuture<Window>>,
	fence_signal_future: Option<
		FenceSignalFuture<
			PresentFuture<
				CommandBufferExecFuture<Box<dyn GpuFuture>, PrimaryAutoCommandBuffer>, 
				Window
			>
		>
	>,
	pub need_new_swapchain: bool
}
impl Swapchain
{
	pub fn new(
		vk_dev: Arc<vulkano::device::Device>, 
		queue: Arc<vulkano::device::Queue>, 
		window_surface: Arc<vulkano::swapchain::Surface<Window>>
	) -> Result<Swapchain, Box<dyn std::error::Error>>
	{
		// query surface capabilities
		let surf_caps = window_surface.capabilities(vk_dev.physical_device())?;

		let (swapchain, swapchain_images) = vulkano::swapchain::Swapchain::start(vk_dev.clone(), window_surface.clone())
			.num_images(surf_caps.min_image_count)
			.format(Format::B8G8R8A8_SRGB)
			.usage(vulkano::image::ImageUsage::color_attachment())
			.sharing_mode(&queue)
			.build()?;

		// create render pass
		let swapchain_rp = vulkano::single_pass_renderpass!(vk_dev.clone(),
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

		let framebuffers = create_framebuffers(&swapchain_images, swapchain_rp.clone())?;

		//let previous_frame_end = Some(sync::now(vk_dev.clone()).boxed());
    	//let rotation_start = std::time::Instant::now();

		Ok(Swapchain{
			vk_dev: vk_dev,
			swapchain: swapchain,
			swapchain_images: swapchain_images,
			swapchain_rp: swapchain_rp,
			framebuffers: framebuffers,
			cur_image_num: 0,
			acquire_future: None,
			//previous_frame_end: previous_frame_end,
			fence_signal_future: None,
			need_new_swapchain: false
		})
	}

	pub fn recreate_swapchain(&mut self) -> Result<(), Box<dyn std::error::Error>>
	{
		let dimensions: [u32; 2] = self.swapchain.surface().window().inner_size().into();

		let (new_swapchain, new_images) = self.swapchain.recreate().dimensions(dimensions).build()?;
		self.swapchain = new_swapchain;
		self.swapchain_images = new_images;
		self.framebuffers = create_framebuffers(&self.swapchain_images, self.swapchain_rp.clone())?;

		self.need_new_swapchain = false;

		Ok(())
	}

	pub fn get_next_image(&mut self) -> Result<Arc<vulkano::render_pass::Framebuffer>, Box<dyn std::error::Error>>
	{
		if self.need_new_swapchain {
			self.recreate_swapchain()?;
		}
		match self.fence_signal_future.as_mut() {
			Some(p) => p.cleanup_finished(),
			None => ()
		}

		let (image_num, suboptimal, acquire_future) =
			match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
				Ok(r) => r,
				Err(vulkano::swapchain::AcquireError::OutOfDate) => {
					// recreate the swapchain then try again
					self.need_new_swapchain = true;
					return self.get_next_image();
				}
				Err(e) => return Err(Box::new(e))
			};

		if suboptimal {
			self.need_new_swapchain = true;
		}

		self.cur_image_num = image_num;
		self.acquire_future = Some(acquire_future);
		
		Ok(self.framebuffers[self.cur_image_num].clone())
	}

	pub fn get_current_image(&self) -> Arc<vulkano::render_pass::Framebuffer>
	{
		self.framebuffers[self.cur_image_num].clone()
	}

	pub fn wait_for_fence(&self) -> Result<(), FlushError>
	{
		match self.fence_signal_future.as_ref() {
			Some(f) => f.wait(None),
			None => Ok(())
		}
	}

	pub fn submit_commands(&mut self, 
		submit_cb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, StandardCommandPoolBuilder>,
		queue: Arc<vulkano::device::Queue>,
		join_futures: std::collections::LinkedList<Box<dyn GpuFuture>>
	) 
		-> Result<(), Box<dyn std::error::Error>>
	{
		let built_cb = submit_cb.build()?;

		let mut joined_future = self.acquire_future.take().ok_or(ImageNotAcquired)?.boxed();
		
		match self.fence_signal_future.take() {
			Some(f) => joined_future = joined_future.join(f).boxed(),
			None => ()
		}

		// join futures from images and buffers being uploaded
		let mut join_count: usize = 0;
		for join_future in join_futures {
			joined_future = joined_future.join(join_future).boxed();
			join_count += 1;
		}
		if join_count > 0 {
			log::debug!("Joined {} futures", join_count);
		}

		let future_result = joined_future
			.then_execute(queue.clone(), built_cb)?
			.then_swapchain_present(queue, self.swapchain.clone(), self.cur_image_num)
			.then_signal_fence_and_flush();

		match future_result {
			Ok(future) => {
				self.fence_signal_future = Some(future);
			}
			Err(FlushError::OutOfDate) => {
				self.recreate_swapchain()?;
			}
			Err(e) => {
				return Err(Box::new(e))
			}
		}

		Ok(())
	}

	pub fn render_pass(&self) -> Arc<vulkano::render_pass::RenderPass> 
	{
		self.swapchain_rp.clone()
	}

	pub fn dimensions(&self) -> [u32; 2]
	{
		self.swapchain.dimensions()
	}

	pub fn surface(&self) -> Arc<vulkano::swapchain::Surface<winit::window::Window>>
	{
		self.swapchain.surface().clone()
	}
}

fn create_framebuffers(
	images: &Vec<Arc<vulkano::image::swapchain::SwapchainImage<Window>>>, 
	render_pass: Arc<vulkano::render_pass::RenderPass>
) -> Result<Vec::<Arc<Framebuffer>>, Box<dyn std::error::Error>>
{
	let mut framebuffers = Vec::<Arc<Framebuffer>>::with_capacity(images.len());
	for img in images.iter() {
		let view = vulkano::image::view::ImageView::new(img.clone())?;
		// TODO: add depth buffers
		framebuffers.push(Framebuffer::start(render_pass.clone()).add(view)?.build()?);
	}
	
	Ok(framebuffers)
}

#[derive(Debug)]
struct ImageNotAcquired;
impl std::error::Error for ImageNotAcquired {}
impl std::fmt::Display for ImageNotAcquired {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "An attempt was made to submit a primary command buffer when no image was acquired!")
    }
}
