/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::swapchain::{
	AcquireError, PresentMode, Surface, SurfaceInfo, SwapchainAcquireFuture, SwapchainCreateInfo, 
	SwapchainPresentInfo
};
use vulkano::sync::{FenceSignalFuture, FlushError, GpuFuture};
use winit::window::Window;

use crate::GenericEngineError;

pub struct Swapchain
{
	window: Arc<Window>,
	swapchain: Arc<vulkano::swapchain::Swapchain>,
	images: Vec<Arc<SwapchainImage>>,

	recreate_pending: bool,

	acquire_future: Option<SwapchainAcquireFuture>,
	submission_future: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,
}
impl Swapchain
{
	pub fn new(vk_dev: Arc<vulkano::device::Device>, window: Window) -> Result<Self, GenericEngineError>
	{
		let window_arc = Arc::new(window);
		let surface = vulkano_win::create_surface_from_winit(window_arc.clone(), vk_dev.instance().clone())?;

		let pd = vk_dev.physical_device();
		let surface_formats = pd.surface_formats(&surface, SurfaceInfo::default())?;
		let surface_caps = pd.surface_capabilities(&surface, SurfaceInfo::default())?;
		log::info!("Available surface format and color space combinations:");
		surface_formats.iter().for_each(|f| log::info!("{:?}", f));

		let image_usage = ImageUsage { transfer_dst: true, ..ImageUsage::empty() };
		let create_info = SwapchainCreateInfo {
			min_image_count: surface_caps.min_image_count,
			image_format: Some(Format::B8G8R8A8_SRGB),
			image_usage,
			present_mode: PresentMode::Immediate,
			..Default::default()
		};
		let (swapchain, images) = vulkano::swapchain::Swapchain::new(vk_dev.clone(), surface, create_info)?;

		Ok(Swapchain {
			window: window_arc,
			swapchain,
			images,
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
		create_info.image_extent = self.window.inner_size().into();
		let (new_swapchain, new_images) = self.swapchain.recreate(create_info)?;
		self.swapchain = new_swapchain;
		self.images = new_images;

		let dimensions_changed = self.swapchain.image_extent() != prev_dimensions;
		if dimensions_changed {
			log::info!("Swapchain resized: {:?} -> {:?}", prev_dimensions, self.swapchain.image_extent());
		}
		Ok(dimensions_changed)
	}

	/// Get the next swapchain image.
	/// Returns the image and a bool indicating if the image dimensions changed.
	/// Subsequent command buffer commands will fail with `vulkano::sync::AccessError::AlreadyInUse`
	/// if this isn't run after every swapchain command submission.
	pub fn get_next_image(&mut self) -> Result<(Arc<SwapchainImage>, bool), GenericEngineError>
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
					log::warn!("Swapchain image is suboptimal, swapchain recreate pending...");
				}
				self.recreate_pending = suboptimal;
				self.acquire_future = Some(acquire_future);
				Ok((self.images[image_num as usize].clone(), dimensions_changed))
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

		let present_info = SwapchainPresentInfo::swapchain_image_index(
			acquire_future.swapchain().clone(), acquire_future.image_index()
		);

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

	pub fn dimensions(&self) -> [u32; 2]
	{
		self.swapchain.image_extent()
	}

	pub fn get_surface(&self) -> Arc<Surface>
	{
		self.swapchain.surface().clone()
	}
}

