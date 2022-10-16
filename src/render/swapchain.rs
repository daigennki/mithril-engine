/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::device::{DeviceOwned, Queue};
use vulkano::format::Format;
use vulkano::image::{attachment::AttachmentImage, view::ImageView, ImageAccess, ImageUsage, SwapchainImage};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{
	AcquireError, PresentInfo, PresentMode, Surface, SurfaceInfo, SwapchainAcquireFuture, SwapchainCreateInfo,
};
use vulkano::sync::{FenceSignalFuture, FlushError, GpuFuture};
use winit::window::Window;

use crate::GenericEngineError;

pub struct Swapchain
{
	swapchain: Arc<vulkano::swapchain::Swapchain<Window>>,
	framebuffers: Vec<Arc<Framebuffer>>,
	recreate_pending: bool,

	acquire_future: Option<SwapchainAcquireFuture<Window>>,
	submission_future: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,
}
impl Swapchain
{
	pub fn new(vk_dev: Arc<vulkano::device::Device>, surface: Arc<Surface<Window>>) -> Result<Self, GenericEngineError>
	{
		let swapchain_rp = vulkano::ordered_passes_renderpass!(
			vk_dev.clone(),
			attachments: {
				color: {
					load: DontCare,	// this is DontCare since drawing the skybox effectively clears the image for us
					store: Store,
					format: Format::B8G8R8A8_SRGB,
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

		let (swapchain, framebuffers) = create_swapchain(vk_dev.clone(), surface, swapchain_rp)?;

		Ok(Swapchain {
			swapchain,
			framebuffers,
			recreate_pending: false,
			acquire_future: None,
			submission_future: None,
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
		self.framebuffers = create_framebuffers(new_images, self.render_pass())?;

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
		if let Some(f) = self.submission_future.as_mut() {
			f.cleanup_finished();
		}

		let dimensions_changed = if self.recreate_pending {
			self.fit_window()? // recreate the swapchain
		} else {
			false
		};

		if self.acquire_future.is_some() {
			panic!("`get_next_image` called when an image has already been acquired without being submitted!");
		}

		match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
			Ok((image_num, suboptimal, acquire_future)) => {
				if suboptimal {
					log::warn!("Image is suboptimal, swapchain recreate pending...");
				}
				self.recreate_pending = suboptimal;
				self.acquire_future = Some(acquire_future);
				Ok((self.framebuffers[image_num].clone(), dimensions_changed))
			}
			Err(AcquireError::OutOfDate) => {
				log::warn!("Swapchain out of date, recreating...");
				self.recreate_pending = true;
				self.get_next_image()
			}
			Err(e) => Err(Box::new(e)),
		}
	}

	/// Submit a primary command buffer's commands.
	/// Optionally, a command buffer `transfers` containing only transfer commands could also be set. It
	/// will be executed before `cb` on the same queue, or on `transfer_queue` if it is also set.
	pub fn submit_commands(
		&mut self, cb: PrimaryAutoCommandBuffer, queue: Arc<Queue>, transfers: Option<PrimaryAutoCommandBuffer>,
		transfer_queue: Option<Arc<Queue>>,
	) -> Result<(), GenericEngineError>
	{
		let acquire_future = self
			.acquire_future
			.take()
			.expect("Command buffer submitted without acquiring an image!");

		let present_info = PresentInfo {
			index: acquire_future.image_id(),
			..PresentInfo::swapchain(acquire_future.swapchain().clone())
		};

		let mut joined_futures = acquire_future.boxed_send_sync();
		if let Some(f) = self.submission_future.take() {
			f.wait(None)?; // wait for the previous submission to finish, to make sure resources are no longer in use
			joined_futures = Box::new(joined_futures.join(f));
		}
		if let Some(t) = transfers {
			joined_futures = joined_futures
				.then_execute(transfer_queue.unwrap_or_else(|| queue.clone()), t)?
				.boxed_send_sync();
		}

		let future_result = joined_futures
			.then_execute(queue.clone(), cb)?
			.then_swapchain_present(queue, present_info)
			.boxed_send_sync()
			.then_signal_fence_and_flush();

		match future_result {
			Ok(future) => self.submission_future = Some(future),
			Err(FlushError::OutOfDate) => {
				log::warn!("Swapchain out of date, recreating...");
				self.recreate_pending = true;
			}
			Err(e) => return Err(Box::new(e)),
		}

		Ok(())
	}

	pub fn render_pass(&self) -> Arc<RenderPass>
	{
		self.framebuffers[0].render_pass().clone()
	}

	pub fn dimensions(&self) -> [u32; 2]
	{
		self.swapchain.image_extent()
	}

	/// Get the currently acquired swapchain image framebuffer.
	/// Returns `None` if no image is currently acquired.
	pub fn get_current_framebuffer(&self) -> Option<Arc<Framebuffer>>
	{
		if let Some(f) = self.acquire_future.as_ref() {
			Some(self.framebuffers[f.image_id()].clone())
		} else {
			None
		}
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
			dimensions: [dim[0] as f32, dim[1] as f32],
			depth_range: 0.0..1.0,
		}
	}
}

fn create_swapchain(
	vk_dev: Arc<vulkano::device::Device>, surface: Arc<Surface<Window>>, render_pass: Arc<RenderPass>,
) -> Result<(Arc<vulkano::swapchain::Swapchain<Window>>, Vec<Arc<Framebuffer>>), GenericEngineError>
{
	let pd = vk_dev.physical_device();
	let surface_formats = pd.surface_formats(&surface, SurfaceInfo::default())?;
	let surface_caps = pd.surface_capabilities(&surface, SurfaceInfo::default())?;

	log::info!("Available surface format and color space combinations:");
	surface_formats.iter().for_each(|f| log::info!("{:?}", f));

	let image_format = Some(render_pass.attachments()[0].format.unwrap());
	let image_usage = ImageUsage { color_attachment: true, ..ImageUsage::empty() };
	let create_info = SwapchainCreateInfo {
		min_image_count: surface_caps.min_image_count,
		image_format,
		image_usage,
		present_mode: PresentMode::Immediate,
		..Default::default()
	};

	let (swapchain, images) = vulkano::swapchain::Swapchain::new(vk_dev.clone(), surface, create_info)?;
	let framebuffers = create_framebuffers(images, render_pass)?;

	Ok((swapchain, framebuffers))
}

fn create_framebuffers(
	images: Vec<Arc<SwapchainImage<Window>>>, render_pass: Arc<RenderPass>,
) -> Result<Vec<Arc<Framebuffer>>, GenericEngineError>
{
	let depth_format = render_pass.attachments()[1].format.unwrap();
	images
		.iter()
		.map(|img| {
			let depth_image = AttachmentImage::new(img.device().clone(), img.dimensions().width_height(), depth_format)?;
			let fb_create_info = FramebufferCreateInfo {
				attachments: vec![
					ImageView::new_default(img.clone())?,
					ImageView::new_default(depth_image)?,
				],
				..Default::default()
			};
			Ok(Framebuffer::new(render_pass.clone(), fb_create_info)?)
		})
		.collect()
}
