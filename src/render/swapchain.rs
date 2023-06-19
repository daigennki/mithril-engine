/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;
use vulkano::command_buffer::{PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::swapchain::{
	AcquireError, PresentMode, Surface, SurfaceInfo, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::{
	future::{FenceSignalFuture, GpuFuture},
	FlushError,
};
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
		let surface = vulkano_win::create_surface_from_handle(window_arc.clone(), vk_dev.instance().clone())?;

		let pd = vk_dev.physical_device();
		let surface_caps = pd.surface_capabilities(&surface, SurfaceInfo::default())?;

		let surface_formats = pd.surface_formats(&surface, SurfaceInfo::default())?;
		log::info!("Available surface format and color space combinations:");
		surface_formats.iter().for_each(|f| log::info!("{:?}", f));

		let surface_present_modes = pd.surface_present_modes(&surface)?;
		log::info!("Available surface present modes: {:?}", Vec::from_iter(surface_present_modes));

		// Explicitly set `image_extent` since some environments, such as Wayland, require it to not cause a panic.
		let image_extent: [u32; 2] = window_arc.inner_size().into();

		// NVIDIA on Linux (possibly only when using Wayland?) only supports B8G8R8A8_UNORM + SrgbNonLinear, so it would be a
		// safer bet than B8G8R8A8_SRGB. B8G8R8A8_UNORM does in fact have slightly wider support than B8G8R8A8_SRGB:
		// https://vulkan.gpuinfo.org/listsurfaceformats.php?platform=linux
		// This means we need to be sure to convert from linear to nonlinear sRGB beforehand. See `RenderContext::submit_frame`
		// for that conversion.
		let create_info = SwapchainCreateInfo {
			min_image_count: surface_caps.min_image_count,
			image_extent,
			image_format: Some(Format::B8G8R8A8_UNORM),
			image_usage: ImageUsage::TRANSFER_DST,
			present_mode: PresentMode::Fifo,
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
	pub fn fit_window(&mut self) -> Result<bool, GenericEngineError>
	{
		let prev_dimensions = self.swapchain.image_extent();
		let mut create_info = self.swapchain.create_info();
		create_info.image_extent = self.window.inner_size().into();
		let (new_swapchain, new_images) = self.swapchain.recreate(create_info)?;
		self.swapchain = new_swapchain;
		self.images = new_images;

		let dimensions_changed = self.swapchain.image_extent() != prev_dimensions;
		if dimensions_changed {
			log::info!(
				"Swapchain resized: {:?} -> {:?}",
				prev_dimensions,
				self.swapchain.image_extent()
			);
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

	pub fn submit_transfer_on_graphics_queue(
		&mut self,
		cb: PrimaryAutoCommandBuffer,
		queue: Arc<Queue>,
	) -> Result<(), GenericEngineError>
	{
		self.submission_future = Some(match self.submission_future.take() {
			Some(f) => cb
				.execute_after(f, queue.clone())?
				.boxed_send_sync()
				.then_signal_fence_and_flush()?,
			None => cb.execute(queue.clone())?.boxed_send_sync().then_signal_fence_and_flush()?,
		});
		Ok(())
	}

	/// Submit a primary command buffer's commands (where the command buffer is expected to manipulate the currently acquired
	/// swapchain image, usually blitting to it) and then present the resulting image.
	/// Optionally, a GpuFuture `after` to wait for (such as for joining submitted transfers on another queue) can be given, so
	/// that graphics operations don't begin until after the future is reached.
	/// Note that `after` does not need to be a signalled fence or semaphore, as signalling will be done in this function.
	pub fn present(
		&mut self,
		cb: PrimaryAutoCommandBuffer,
		queue: Arc<Queue>,
		after: Option<Box<dyn GpuFuture + Send + Sync>>,
	) -> Result<(), GenericEngineError>
	{
		let acquire_future = self
			.acquire_future
			.take()
			.expect("Command buffer submitted without acquiring an image!");

		let present_info =
			SwapchainPresentInfo::swapchain_image_index(acquire_future.swapchain().clone(), acquire_future.image_index());

		let mut joined_futures = acquire_future.boxed_send_sync();

		if let Some(f) = after {
			// Ideally we'd use a semaphore instead of a fence here, but apprently it's borked in Vulkano right now.
			let fence_future = f.then_signal_fence();
			fence_future.wait(None)?;
			joined_futures = Box::new(joined_futures.join(fence_future));
		}

		if let Some(f) = self.submission_future.take() {
			f.wait(None)?; // wait for the previous submission to finish, to make sure resources are no longer in use
			joined_futures = Box::new(joined_futures.join(f));
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
