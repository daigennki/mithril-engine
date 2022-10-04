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
use vulkano::sync::{ FlushError, GpuFuture, FenceSignalFuture,  };
use vulkano::swapchain::{ SwapchainCreateInfo, SurfaceInfo, Surface, AcquireError, PresentFuture };
use vulkano::image::{ SwapchainImage, ImageAccess, ImageUsage, attachment::AttachmentImage, view::ImageView };
use vulkano::pipeline::graphics::viewport::Viewport;

use crate::GenericEngineError;

pub struct Swapchain
{
	swapchain: Arc<vulkano::swapchain::Swapchain<Window>>,
	swapchain_rp: Arc<RenderPass>,
	framebuffers: Vec<Arc<Framebuffer>>,
	cur_image_num: usize,

	// The future to wait for before the next submission.
	// Includes image acquisition, as well as the previous frame's fence signal.
	wait_before_submit: Option<Box<dyn GpuFuture + Send + Sync>>,
	fence_signal_future: Option<FenceSignalFuture<PresentFuture<CommandBufferExecFuture<Box<dyn GpuFuture + Send + Sync>, PrimaryAutoCommandBuffer>, winit::window::Window>>>
}
impl Swapchain
{
	pub fn new(vk_dev: Arc<vulkano::device::Device>, surface: Arc<Surface<Window>>) -> Result<Self, GenericEngineError>
	{
		let (swapchain, swapchain_images) = create_swapchain(vk_dev.clone(), surface)?;
		let swapchain_rp = vulkano::ordered_passes_renderpass!(
			vk_dev.clone(),
			attachments: {
				color: {
					load: DontCare,	// this is DontCare since drawing the skybox effectively clears the image for us
					store: Store,
					format: swapchain.image_format(),
					samples: 1,
				},
				depth: {
					load: DontCare,	// this too is DontCare since the skybox clears it with 1.0
					store: Store,	// order-independent transparency might need this to be `Store`
					format: Format::D16_UNORM,	// NOTE: 24-bit depth formats are unsupported on a significant number of GPUs
					samples: 1,
				}
			}, 
			passes: [
				{	// general rendering subpass
					color: [color],
					depth_stencil: {depth},
					input: []
				},
				{	// egui subpass
					color: [color],
					depth_stencil: {},
					input: []
				}
			]
		)?;

		Ok(Swapchain{
			swapchain: swapchain,
			swapchain_rp: swapchain_rp.clone(),
			framebuffers: create_framebuffers(swapchain_images, swapchain_rp)?,
			cur_image_num: 0,
			wait_before_submit: None,
			fence_signal_future: None
		})
	}

	/// Recreate the swapchain to fit the window's requirements (e.g., window size changed).
	/// Returns `Ok(true)` if the image extent has changed.
	fn fit_window(&mut self) -> Result<bool, GenericEngineError>
	{
		let prev_dimensions = self.swapchain.image_extent();
		let mut create_info = self.swapchain.create_info();
		create_info.image_extent = self.swapchain.surface().window().inner_size().into();
		let (new_swapchain, new_images) = self.swapchain.recreate(create_info)?;
		self.swapchain = new_swapchain;
		self.framebuffers = create_framebuffers(new_images, self.swapchain_rp.clone())?;
		
		let dimensions_changed = self.swapchain.image_extent() != prev_dimensions;
		if dimensions_changed {
			log::info!("Swapchain resized: {:?} -> {:?}", prev_dimensions, self.swapchain.image_extent());
		}
		Ok(dimensions_changed)
	}

	/// Get the next swapchain image.
	/// Returns the corresponding framebuffer, and a bool indicating if the image dimensions changed.
	/// Subsequent command buffer commands will fail with `vulkano::sync::AccessError::AlreadyInUse`
	/// if this isn't run after every swapchain command submission.
	pub fn get_next_image(&mut self) -> Result<(Arc<Framebuffer>, bool), GenericEngineError>
	{
		// clean up resources from finished submissions
		self.wait_before_submit.as_mut().map(GpuFuture::cleanup_finished);
		self.fence_signal_future.as_mut().map(GpuFuture::cleanup_finished);

		match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
			Ok((image_num, false, acquire_future)) => {
				self.cur_image_num = image_num;
				self.wait_before_submit = Some(
					match self.wait_before_submit.take() {
						Some(f) => Box::new(f.join(acquire_future)),
						None => Box::new(acquire_future)
					}
				);
				Ok((self.framebuffers[self.cur_image_num].clone(), false))
			},
			Ok((_, true, _)) |	// if suboptimal
			Err(AcquireError::OutOfDate) => {
				log::info!("Swapchain out of date or suboptimal, recreating...");
				let dimensions_changed = self.fit_window()?;	// recreate the swapchain...
				let (fb, dim_changed_again) = self.get_next_image()?;	// ...then try again
				Ok((fb, (dimensions_changed || dim_changed_again)))
			},
			Err(e) => Err(Box::new(e))
		}
	}

	/// Submit a primary command buffer's commands.
	pub fn submit_commands(
		&mut self, cb: PrimaryAutoCommandBuffer, queue: Arc<Queue>, futures: Option<Box<dyn GpuFuture + Send + Sync>>
	)
		-> Result<(), GenericEngineError>
	{
		let mut joined_futures = self.wait_before_submit.take().ok_or(NoSubmitFuturesError)?.boxed_send_sync();
		if let Some(f) = futures {
			// join the joined futures from images and buffers being uploaded
			joined_futures = Box::new(joined_futures.join(f));
		}
		if let Some(f) = self.fence_signal_future.take() {
			joined_futures = Box::new(joined_futures.join(f));
		}

		let future_result = joined_futures
			.then_execute(queue.clone(), cb)?
			.then_swapchain_present(queue, self.swapchain.clone(), self.cur_image_num)
			.then_signal_fence_and_flush();

		match future_result {
			Ok(future) => self.fence_signal_future = Some(future),
			Err(FlushError::OutOfDate) => (),	// let `get_next_image` detect the error next frame
			Err(e) => return Err(Box::new(e))
		}

		Ok(())
	}

	pub fn wait_for_fence(&self) -> Result<(), FlushError>
	{
		self.fence_signal_future.as_ref().map(|f| f.wait(None)).unwrap_or(Ok(()))
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

	pub fn get_surface(&self) -> Arc<Surface<Window>>
	{
		self.swapchain.surface().clone()
	}
	
	/// Get a viewport that fills the entire current swapchain image.
	pub fn get_viewport(&self) -> Viewport
	{
		let dim = self.dimensions();
		Viewport {
			origin: [0.0, 0.0], 
			dimensions: [ dim[0] as f32, dim[1] as f32 ],
			depth_range: 0.0..1.0 
		}	
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
	images.iter().map(|img| {
		let depth_image = AttachmentImage::new(img.device().clone(), img.dimensions().width_height(), depth_format)?;
		let fb_create_info = vulkano::render_pass::FramebufferCreateInfo {
			attachments: vec![ ImageView::new_default(img.clone())?, ImageView::new_default(depth_image)? ],
			..Default::default()
		};
		Ok(Framebuffer::new(render_pass.clone(), fb_create_info)?)
	}).collect()
}

#[derive(Debug)]
struct NoSubmitFuturesError;
impl std::error::Error for NoSubmitFuturesError {}
impl std::fmt::Display for NoSubmitFuturesError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "`wait_before_submit` was `None`; did you forget to call `get_next_image`?")
    }
}

