/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::{Arc, Mutex};

use vulkano::buffer::{
	allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
	BufferContents, BufferUsage, Subbuffer,
};
use vulkano::command_buffer::{
	allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BufferImageCopy,
	CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
};
use vulkano::device::Queue;
use vulkano::image::{Image, ImageSubresourceLayers};
use vulkano::memory::{
	allocator::{DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator},
	DeviceAlignment,
};
use vulkano::sync::{future::FenceSignalFuture, GpuFuture};
use vulkano::DeviceSize;

const STAGING_ARENA_SIZE: DeviceSize = 64 * 1024 * 1024;

pub struct TransferManager
{
	// Transfers to initialize buffers and images. If there is an asynchronous transfer queue,
	// these will be performed while the CPU is busy with building the draw command buffers.
	// Otherwise, these will be run at the beginning of the next graphics submission, before draws
	// are performed.
	async_transfers: Vec<Box<dyn StagingWorkTrait>>,
	staging_buffer_allocator: Mutex<SubbufferAllocator>,
	transfer_queue: Option<Arc<Queue>>,
	transfer_future: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,

	// Buffer updates to run at the beginning of the next graphics submission.
	buffer_updates: Option<Box<dyn UpdateBufferDataTrait>>,
}
impl TransferManager
{
	pub fn new(transfer_queue: Option<Arc<Queue>>, memory_allocator: Arc<StandardMemoryAllocator>) -> Self
	{
		let pool_create_info = SubbufferAllocatorCreateInfo {
			arena_size: STAGING_ARENA_SIZE,
			buffer_usage: BufferUsage::TRANSFER_SRC,
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buffer_allocator = Mutex::new(SubbufferAllocator::new(memory_allocator.clone(), pool_create_info));

		Self {
			async_transfers: Vec::with_capacity(256),
			staging_buffer_allocator,
			transfer_queue,
			transfer_future: Default::default(),
			buffer_updates: None,
		}
	}

	fn add_transfer<T>(&mut self, work: StagingWork<T>)
	where
		T: BufferContents + Copy
	{
		// If adding this transfer would cause staging buffer usage to exceed the staging buffer
		// arena size, submit transfers immediately before adding the transfer.
		if let Some(pending_transfer_size) = self.pending_transfer_size() {
			let transfer_layout = work.device_layout();
			let (extended_layout, _) = pending_transfer_size.extend(transfer_layout).unwrap();

			if extended_layout.size() > STAGING_ARENA_SIZE {
				log::debug!("exceeded staging buffer arena size, submitting previous transfers now...");

				todo!();
			}
		}

		self.async_transfers.push(Box::new(work));
	}

	/// Add staging work for a new buffer.
	/// If there is a separate transfer queue, this will be performed asynchronously with the CPU
	/// building the draw command buffers.
	///
	/// If the buffer might be in use by a previous buffer, use `update_buffer` instead.
	pub fn copy_to_buffer<T>(&mut self, data: Vec<T>, dst_buf: Subbuffer<[T]>)
	where
		T: BufferContents + Copy,
	{
		assert!(!data.is_empty());

		let work = StagingWork {
			data,
			dst: StagingDst::Buffer(dst_buf),
		};
		self.add_transfer(work);
	}

	/// Add staging work for a new image.
	/// If there is a separate transfer queue, this will be performed asynchronously with the CPU
	/// building the draw command buffers.
	pub fn copy_to_image<Px>(&mut self, data: Vec<Px>, dst_image: Arc<Image>)
	where
		Px: BufferContents + Copy,
	{
		assert!(!data.is_empty());
		assert!(dst_image.format().block_size().is_power_of_two());

		let work = StagingWork {
			data,
			dst: StagingDst::Image(dst_image),
		};
		self.add_transfer(work);
	}

	/// Update a buffer at the begninning of the next graphics submission.
	pub fn update_buffer<T>(&mut self, data: Box<[T]>, dst_buf: Subbuffer<[T]>)
	where
		T: BufferContents + Copy,
	{
		assert!(data.len() > 0);

		let buf_update = UpdateBufferData {
			dst_buf,
			data,
			next: self.buffer_updates.take(),
		};
		self.buffer_updates = Some(Box::new(buf_update));
	}

	fn submit_transfers(
		&mut self,
		command_buffer_allocator: &StandardCommandBufferAllocator,
		queue: Arc<Queue>,
	) -> crate::Result<()>
	{
		if !self.async_transfers.is_empty() {
			let mut cb = AutoCommandBufferBuilder::primary(
				command_buffer_allocator,
				queue.queue_family_index(),
				CommandBufferUsage::OneTimeSubmit,
			)?;

			self.add_copies(&mut cb)?;

			let transfer_future = if let Some(f) = self.transfer_future.take() {
				f.wait(None)?;

				cb.build()?
					.execute_after(f, queue)
					.unwrap()
					.boxed_send_sync()
					.then_signal_fence_and_flush()?
			} else {
				cb.build()?
					.execute(queue)
					.unwrap()
					.boxed_send_sync()
					.then_signal_fence_and_flush()?
			};

			self.transfer_future = Some(transfer_future);
		}

		Ok(())
	}

	/// Submit the asynchronous transfers that are waiting.
	/// Run this just before beginning to build the draw command buffers,
	/// so that the transfers can be done while the CPU is busy with building the draw command buffers.
	///
	/// This does nothing if there is no asynchronous transfer queue. In such a case, the transfers will
	/// instead be done at the beginning of the graphics submission on the graphics queue.
	pub fn submit_async_transfers(&mut self, command_buffer_allocator: &StandardCommandBufferAllocator) -> crate::Result<()>
	{
		if let Some(q) = self.transfer_queue.clone() {
			self.submit_transfers(command_buffer_allocator, q)?;
		}

		Ok(())
	}

	pub fn add_synchronous_transfer_commands(
		&mut self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
	) -> crate::Result<()>
	{
		// buffer updates
		let mut buf_update_option = self.buffer_updates.take();
		while let Some(buf_update) = buf_update_option.take() {
			buf_update_option = buf_update.add_command(cb);
		}

		// If there is no asynchronous transfer queue, do them on the graphics queue.
		if self.transfer_queue.is_none() {
			self.add_copies(cb)?;
		}

		Ok(())
	}

	/// Calculate the total staging buffer usage for the pending transfers.
	///
	/// Returns `None` if there are no pending transfers.
	fn pending_transfer_size(&self) -> Option<DeviceLayout>
	{
		self.async_transfers
			.iter()
			.filter(|work| !work.will_use_update_buffer())
			.map(|work| work.device_layout())
			.reduce(|acc, device_layout| acc.extend(device_layout).unwrap().0)
	}

	fn add_copies(&mut self, cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> crate::Result<()>
	{
		// TODO: If usage exceeds a certain threshold, defer further transfers to the next submission.
		let staging_buf_usage_frame = self.pending_transfer_size();

		let mut staging_buf_alloc_guard = self.staging_buffer_allocator.lock().unwrap();

		let staging_buf_usage_bytes = staging_buf_usage_frame.map(|usage| usage.size()).unwrap_or(0);
		assert!(staging_buf_usage_bytes <= staging_buf_alloc_guard.arena_size());

		for work in self.async_transfers.drain(..) {
			work.add_command(cb, &mut staging_buf_alloc_guard)?;
		}

		Ok(())
	}

	pub fn take_transfer_future(&mut self) -> Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>
	{
		self.transfer_future.take()
	}
}

/// Synchronous buffer updates that must be submitted on the graphics queue. This is a singly
/// linked list, and the next staging work is returned by `add_command`.
struct UpdateBufferData<T: BufferContents + Copy>
{
	dst_buf: Subbuffer<[T]>,
	data: Box<[T]>,
	next: Option<Box<dyn UpdateBufferDataTrait>>,
}
impl<T: BufferContents + Copy> UpdateBufferDataTrait for UpdateBufferData<T>
{
	fn add_command(
		self: Box<Self>,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
	) -> Option<Box<dyn UpdateBufferDataTrait>>
	{
		cb_builder.update_buffer(self.dst_buf, self.data).unwrap();
		self.next
	}
}
trait UpdateBufferDataTrait: Send + Sync
{
	fn add_command(
		self: Box<Self>,
		_: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
	) -> Option<Box<dyn UpdateBufferDataTrait>>;
}

/// Staging work that may be submitted to either the asynchronous transfer queue or the graphics
/// queue.
///
/// # Note for when this is used with images
///
/// `data` here will be used to initialize the full extent of all mipmap levels and array layers
/// (in that order). The data must be tightly packed together. For example, bitmap data for an
/// image with 2 mip levels and 2 array layers must be structured like:
///
/// - mip level 0
///   - array layer 0
///   - array layer 1
/// - mip level 1
///   - array layer 0
///   - array layer 1
///
struct StagingWork<T: BufferContents + Copy>
{
	data: Vec<T>,
	dst: StagingDst<T>,
}
impl<T: BufferContents + Copy> StagingWorkTrait for StagingWork<T>
{
	fn device_layout(&self) -> DeviceLayout
	{
		let data_size_bytes: DeviceSize = (self.data.len() * std::mem::size_of::<T>()).try_into().unwrap();
		let alignment = self.dst.alignment();
		DeviceLayout::new(data_size_bytes.try_into().unwrap(), alignment).unwrap()
	}

	// If the data is small enough, this will use `update_buffer` instead.
	fn will_use_update_buffer(&self) -> bool
	{
		match self.dst {
			StagingDst::Buffer(_) => {
				let data_size_bytes = std::mem::size_of::<T>() * self.data.len();
				data_size_bytes <= 65536
			}
			_ => false,
		}
	}

	/// Add the command for this staging work.
	fn add_command(
		self: Box<Self>,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		subbuffer_allocator: &mut SubbufferAllocator,
	) -> crate::Result<()>
	{
		// We allocate a subbuffer using a `DeviceLayout` here so that it's aligned to the block
		// size of the format.
		let device_layout = self.device_layout();

		let use_update_buffer = self.will_use_update_buffer();

		match self.dst {
			StagingDst::Buffer(dst_buf) => {
				if use_update_buffer {
					cb_builder.update_buffer(dst_buf, self.data.into_boxed_slice())?;
				} else {
					let staging_buf: Subbuffer<[T]> = subbuffer_allocator.allocate(device_layout)?.reinterpret();
					staging_buf.write().unwrap().copy_from_slice(&self.data);
					cb_builder.copy_buffer(CopyBufferInfo::buffers(staging_buf, dst_buf))?;
				}
			}
			StagingDst::Image(dst_image) => {
				let format = dst_image.format();
				let extent = dst_image.extent();
				let mip_levels = dst_image.mip_levels();
				let array_layers = dst_image.array_layers();

				// generate copies for every mipmap level
				let mut regions = Vec::with_capacity(mip_levels as usize);
				let mut mip_width = extent[0];
				let mut mip_height = extent[1];
				let mut buffer_offset: DeviceSize = 0;
				for mip_level in 0..mip_levels {
					regions.push(BufferImageCopy {
						buffer_offset,
						image_subresource: ImageSubresourceLayers {
							mip_level,
							..ImageSubresourceLayers::from_parameters(format, array_layers)
						},
						image_extent: [mip_width, mip_height, 1],
						..Default::default()
					});

					buffer_offset += super::get_mip_size(format, mip_width, mip_height) * (array_layers as DeviceSize);
					mip_width /= 2;
					mip_height /= 2;
				}

				let staging_buf: Subbuffer<[T]> = subbuffer_allocator.allocate(device_layout)?.reinterpret();
				staging_buf.write().unwrap().copy_from_slice(&self.data);

				let copy_info = CopyBufferToImageInfo {
					regions: regions.into(),
					..CopyBufferToImageInfo::buffer_image(staging_buf, dst_image)
				};
				cb_builder.copy_buffer_to_image(copy_info)?;
			}
		}

		Ok(())
	}
}
trait StagingWorkTrait: Send + Sync
{
	fn device_layout(&self) -> DeviceLayout;

	fn will_use_update_buffer(&self) -> bool;

	fn add_command(
		self: Box<Self>,
		_: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		_: &mut SubbufferAllocator,
	) -> crate::Result<()>;
}

enum StagingDst<T: BufferContents + Copy>
{
	Buffer(Subbuffer<[T]>),
	Image(Arc<Image>),
}
impl<T: BufferContents + Copy> StagingDst<T>
{
	fn alignment(&self) -> DeviceAlignment
	{
		match self {
			Self::Buffer(_) => T::LAYOUT.alignment(),
			Self::Image(image) => image.format().block_size().try_into().unwrap(),
		}
	}
}
