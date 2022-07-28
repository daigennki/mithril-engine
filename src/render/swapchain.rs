/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use winit::window::Window;
use vulkano::device::{ Queue, DeviceOwned };
use vulkano::command_buffer::{ PrimaryAutoCommandBuffer, CommandBufferExecFuture };
use vulkano::format::Format;
use vulkano::render_pass::{ RenderPass, Framebuffer };
use vulkano::sync::{ FlushError, GpuFuture, FenceSignalFuture};
use vulkano::swapchain::{ Surface, AcquireError, SwapchainAcquireFuture, PresentFuture };
use vulkano::image::{ ImageAccess, attachment::AttachmentImage, view::{ ImageViewCreateInfo, ImageView } };

pub struct Swapchain
{
	swapchain: Arc<vulkano::swapchain::Swapchain<Window>>,
	swapchain_rp: Arc<RenderPass>,
	framebuffers: Vec<Arc<Framebuffer>>,
	cur_image_num: usize,
	acquire_future: Option<SwapchainAcquireFuture<Window>>,
	fence_signal_future: Option<
		FenceSignalFuture<PresentFuture<CommandBufferExecFuture<Box<dyn GpuFuture>, PrimaryAutoCommandBuffer>, Window>>
	>,	// sheesh, that's a mouthful
	need_new_swapchain: bool,
	create_info: vulkano::swapchain::SwapchainCreateInfo
}
impl Swapchain
{
	pub fn new(vk_dev: Arc<vulkano::device::Device>, window_surface: Arc<Surface<Window>>) 
		-> Result<Swapchain, Box<dyn std::error::Error>>
	{
		// query surface capabilities
		let surf_caps = vk_dev.physical_device().surface_capabilities(
			&window_surface, vulkano::swapchain::SurfaceInfo::default()
		)?;

		let swapchain_create_info = vulkano::swapchain::SwapchainCreateInfo {
			min_image_count: surf_caps.min_image_count,
			image_format: Some(Format::B8G8R8A8_SRGB),
			image_usage: vulkano::image::ImageUsage::color_attachment(),
			..vulkano::swapchain::SwapchainCreateInfo::default()
		};
		// TODO: sharing mode using `&queue`?
		let (swapchain, swapchain_images) = vulkano::swapchain::Swapchain::new(
			vk_dev.clone(), window_surface.clone(), swapchain_create_info.clone()
		)?;

		// create render pass
		let swapchain_rp = vulkano::single_pass_renderpass!(vk_dev.clone(),
			attachments: {
				color: {
					load: Clear,
					store: Store,
					format: swapchain.image_format(),
					samples: 1,
				},
				depth: {
					load: Clear,
					store: DontCare,
					format: Format::D24_UNORM_S8_UINT,
					samples: 1,
				}
			}, 
			pass: {
				color: [color],
				depth_stencil: {depth}
			}
		)?;

		let framebuffers = create_framebuffers(swapchain_images, swapchain_rp.clone())?;

		Ok(Swapchain{
			swapchain: swapchain,
			swapchain_rp: swapchain_rp,
			framebuffers: framebuffers,
			cur_image_num: 0,
			acquire_future: None,
			fence_signal_future: None,
			need_new_swapchain: false,
			create_info: swapchain_create_info
		})
	}

	/// Get the next swapchain image.
	/// Returns the corresponding framebuffer, and a bool indicating if the image dimensions changed.
	pub fn get_next_image(&mut self) -> Result<(Arc<vulkano::render_pass::Framebuffer>, bool), Box<dyn std::error::Error>>
	{
		// Recreate the swapchain if needed.
		let dimensions_changed = match self.need_new_swapchain {
			true => {
				let prev_dimensions = self.swapchain.image_extent();
				let (new_swapchain, new_images) = self.swapchain.recreate(self.create_info.clone())?;
				self.swapchain = new_swapchain;
				self.framebuffers = create_framebuffers(new_images, self.swapchain_rp.clone())?;
				self.swapchain.image_extent() != prev_dimensions
			}
			false => false
		};
		self.need_new_swapchain = false;
		
		match self.fence_signal_future.as_mut() {
			Some(p) => p.cleanup_finished(),
			None => ()
		}

		let (image_num, suboptimal, acquire_future) =
			match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
				Ok(r) => r,
				Err(AcquireError::OutOfDate) => {
					log::debug!("Swapchain out of date, recreating...");
					self.need_new_swapchain = true;
					return self.get_next_image();	// recreate the swapchain then try again
				}
				Err(e) => return Err(Box::new(e))
			};

		if suboptimal {
			log::debug!("Swapchain is suboptimal, recreating it in next frame...");
			self.need_new_swapchain = true;
		}

		self.cur_image_num = image_num;
		self.acquire_future = Some(acquire_future);
		
		Ok((self.framebuffers[self.cur_image_num].clone(), dimensions_changed))
	}

	/*pub fn get_current_image(&self) -> Arc<vulkano::render_pass::Framebuffer>
	{
		self.framebuffers[self.cur_image_num].clone()
	}

	pub fn wait_for_fence(&self) -> Result<(), FlushError>
	{
		match self.fence_signal_future.as_ref() {
			Some(f) => f.wait(None),
			None => Ok(())
		}
	}*/

	pub fn submit_commands(&mut self, cb: PrimaryAutoCommandBuffer, queue: Arc<Queue>, futures: Option<Box<dyn GpuFuture>>)
		-> Result<(), Box<dyn std::error::Error>>
	{
		let acquire_future = self.acquire_future.take().ok_or(ImageNotAcquired)?;
		let mut joined_future = match self.fence_signal_future.take() {
			Some(f) => f.join(acquire_future).boxed(),
			None => acquire_future.boxed()
		};

		// join the joined futures from images and buffers being uploaded
		match futures {
			Some(f) => joined_future = joined_future.join(f).boxed(),
			None => ()
		}

		let future_result = joined_future
			.then_execute(queue.clone(), cb)?
			.then_swapchain_present(queue, self.swapchain.clone(), self.cur_image_num)
			.then_signal_fence_and_flush();

		match future_result {
			Ok(future) => self.fence_signal_future = Some(future),
			Err(FlushError::OutOfDate) => self.need_new_swapchain = true,
			Err(e) => return Err(Box::new(e))
		}

		Ok(())
	}

	pub fn render_pass(&self) -> Arc<vulkano::render_pass::RenderPass> 
	{
		self.swapchain_rp.clone()
	}

	pub fn dimensions(&self) -> [u32; 2]
	{
		self.swapchain.image_extent()
	}
}

fn create_framebuffers(
	images: Vec<Arc<vulkano::image::swapchain::SwapchainImage<Window>>>, 
	render_pass: Arc<vulkano::render_pass::RenderPass>
) -> Result<Vec::<Arc<Framebuffer>>, Box<dyn std::error::Error>>
{
	let mut framebuffers = Vec::<Arc<Framebuffer>>::with_capacity(images.len());
	for img in images {
		let view_create_info = ImageViewCreateInfo::from_image(&img);
		let view = ImageView::new(img.clone(), view_create_info)?;

		let img_dim = img.dimensions();
		let depth_image = AttachmentImage::new(
			img.device().clone(),
			[ img_dim.width(), img_dim.height() ],
			Format::D24_UNORM_S8_UINT
		)?;
		let depth_view_create_info = ImageViewCreateInfo::from_image(&depth_image);
		let depth_view = ImageView::new(depth_image, depth_view_create_info)?; 

		let fb_create_info = vulkano::render_pass::FramebufferCreateInfo {
			attachments: vec![ view, depth_view ],
			..Default::default()
		};
		framebuffers.push(Framebuffer::new(render_pass.clone(), fb_create_info)?);
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
