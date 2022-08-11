/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use winit::window::Window;
use vulkano::device::{ Queue, DeviceOwned };
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::format::Format;
use vulkano::render_pass::{ RenderPass, Framebuffer };
use vulkano::sync::{ FlushError, GpuFuture };
use vulkano::swapchain::{ SwapchainCreateInfo, SurfaceInfo, Surface, AcquireError };
use vulkano::image::{ SwapchainImage, ImageAccess, ImageUsage, attachment::AttachmentImage, view::ImageView };

use crate::GenericEngineError;

pub struct Swapchain
{
	swapchain: Arc<vulkano::swapchain::Swapchain<Window>>,
	swapchain_rp: Arc<RenderPass>,
	framebuffers: Vec<Arc<Framebuffer>>,
	cur_image_num: usize,
	need_new_swapchain: bool,

	// The future to wait for before the next submission.
	// Includes image acquisition, as well as the previous frame's fence signal. 
	wait_before_submit: Option<Box<dyn GpuFuture + Send + Sync>>,
}
impl Swapchain
{
	pub fn new(vk_dev: Arc<vulkano::device::Device>, surface: Arc<Surface<Window>>) -> Result<Self, GenericEngineError>
	{
		let (swapchain, swapchain_images) = create_swapchain(vk_dev.clone(), surface)?;
		let swapchain_rp = vulkano::single_pass_renderpass!(vk_dev.clone(),
			attachments: {
				color: {
					load: Clear,	// this could be DontCare once we have a skybox set up
					store: Store,
					format: swapchain.image_format(),
					samples: 1,
				},
				depth: {
					load: Clear,
					store: Store,	// order-independent transparency might need this to be `Store`
					format: Format::D16_UNORM,	// 24-bit depth formats are unsupported on a significant number of GPUs
					samples: 1,
				}
			}, 
			pass: {
				color: [color],
				depth_stencil: {depth}
			}
		)?;

		Ok(Swapchain{
			swapchain: swapchain,
			swapchain_rp: swapchain_rp.clone(),
			framebuffers: create_framebuffers(swapchain_images, swapchain_rp)?,
			cur_image_num: 0,
			need_new_swapchain: false,
			wait_before_submit: Some(Box::new(vulkano::sync::now(vk_dev))),
		})
	}

	/// Get the next swapchain image.
	/// Returns the corresponding framebuffer, and a bool indicating if the image dimensions changed.
	/// Subsequent command buffer commands will fail with `vulkano::sync::AccessError::AlreadyInUse`
	/// if this isn't run after every swapchain command submission.
	pub fn get_next_image(&mut self) -> Result<(Arc<Framebuffer>, bool), GenericEngineError>
	{
		// Recreate the swapchain if needed.
		let dimensions_changed = if self.need_new_swapchain {
			let prev_dimensions = self.swapchain.image_extent();
			let (new_swapchain, new_images) = self.swapchain.recreate(self.swapchain.create_info())?;
			self.swapchain = new_swapchain;
			self.framebuffers = create_framebuffers(new_images, self.swapchain_rp.clone())?;
			self.need_new_swapchain = false;
			self.swapchain.image_extent() != prev_dimensions
		} else {
			false
		};
		
		self.wait_before_submit.as_mut().map(|f| f.cleanup_finished());	// clean up resources from finished submissions

		match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
			Ok((image_num, false, acquire_future)) => {
				self.cur_image_num = image_num;
				self.wait_before_submit = Some(
					self.wait_before_submit.take()
						.ok_or("`wait_before_submit` was None")?
						.join(acquire_future)
						.boxed_send_sync()
				);
			},
			Ok((_, true, _)) |	// if suboptimal
			Err(AcquireError::OutOfDate) => {
				log::info!("Swapchain out of date or suboptimal, recreating...");
				self.need_new_swapchain = true;
				return self.get_next_image();	// recreate the swapchain, then try again
			}
			Err(e) => return Err(Box::new(e)),
		};
		
		Ok((self.framebuffers[self.cur_image_num].clone(), dimensions_changed))
	}

	/// Submit a primary command buffer's commands.
	pub fn submit_commands(
		&mut self, cb: PrimaryAutoCommandBuffer, queue: Arc<Queue>, futures: Option<Box<dyn GpuFuture + Send + Sync>>
	)
		-> Result<(), GenericEngineError>
	{
		let mut joined_futures = self.wait_before_submit.take().ok_or("`wait_before_submit` was None")?;
		if let Some(f) = futures {
			// join the joined futures from images and buffers being uploaded
			joined_futures = Box::new(joined_futures.join(f));
		}
		let future_result = joined_futures
			.then_execute(queue.clone(), cb)?
			.then_swapchain_present(queue, self.swapchain.clone(), self.cur_image_num)
			.then_signal_fence_and_flush();

		match future_result {
			Ok(future) => self.wait_before_submit = Some(Box::new(future)),
			Err(FlushError::OutOfDate) => self.need_new_swapchain = true,
			Err(e) => return Err(Box::new(e))
		}

		Ok(())
	}

	pub fn render_pass(&self) -> Arc<RenderPass>
	{
		self.swapchain_rp.clone()
	}

	pub fn dimensions(&self) -> [u32; 2]
	{
		self.swapchain.image_extent()
	}

	pub fn get_current_framebuffer(&self) -> Arc<Framebuffer>
	{
		self.framebuffers[self.cur_image_num].clone()
	}
}

fn create_swapchain(vk_dev: Arc<vulkano::device::Device>, surface: Arc<Surface<Window>>)
	-> Result<(Arc<vulkano::swapchain::Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>), GenericEngineError>
{
	let surface_formats = vk_dev.physical_device().surface_formats(&surface, SurfaceInfo::default())?;
	log::info!("Available surface format and color space combinations:");
	surface_formats.iter().for_each(|f| log::info!("{:?}", f));

	let surface_caps = vk_dev.physical_device().surface_capabilities(&surface, SurfaceInfo::default())?;
	let create_info = SwapchainCreateInfo{
		min_image_count: surface_caps.min_image_count,
		image_format: Some(Format::B8G8R8A8_SRGB),
		image_usage: ImageUsage::color_attachment(),
		..Default::default()
	};

	Ok(vulkano::swapchain::Swapchain::new(vk_dev.clone(), surface, create_info)?)
}

fn create_framebuffers(images: Vec<Arc<SwapchainImage<Window>>>, render_pass: Arc<RenderPass>)
	-> Result<Vec::<Arc<Framebuffer>>, GenericEngineError>
{
	let depth_format = render_pass.attachments().last().unwrap().format.unwrap();
	let mut framebuffers = Vec::<Arc<Framebuffer>>::with_capacity(images.len());
	for img in images {
		let view = ImageView::new_default(img.clone())?;
		let depth_image = AttachmentImage::new(img.device().clone(), img.dimensions().width_height(), depth_format)?;
		let depth_view = ImageView::new_default(depth_image)?;
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

