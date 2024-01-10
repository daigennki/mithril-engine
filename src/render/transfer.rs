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
	allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BufferImageCopy, CommandBufferExecFuture,
	CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
};
use vulkano::device::Queue;
use vulkano::image::Image;
use vulkano::memory::allocator::{DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::{
	future::{FenceSignalFuture, NowFuture},
	GpuFuture,
};
use vulkano::DeviceSize;

pub struct TransferManager
{
	staging_buffer_allocator: Mutex<SubbufferAllocator>,

	// Transfers to initialize buffers and images. If there is an asynchronous transfer queue,
	// these will be performed while the CPU is busy with building the draw command buffers.
	// Otherwise, these will be run at the beginning of the next graphics submission, before draws
	// are performed.
	async_transfers: Vec<Box<dyn StagingWorkTrait>>,

	transfer_queue: Option<Arc<Queue>>,
	transfer_future: Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>,

	// Buffer updates to run at the beginning of the next graphics submission.
	buffer_updates: Vec<Box<dyn UpdateBufferDataTrait>>,
}
impl TransferManager
{
	pub fn new(transfer_queue: Option<Arc<Queue>>, memory_allocator: Arc<StandardMemoryAllocator>) -> Self
	{
		let pool_create_info = SubbufferAllocatorCreateInfo {
			buffer_usage: BufferUsage::TRANSFER_SRC,
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buffer_allocator = Mutex::new(SubbufferAllocator::new(memory_allocator.clone(), pool_create_info));

		// the capacity of these should be adjusted based on number of transfers that might be done in one frame
		let async_transfers = Vec::with_capacity(200);
		let buffer_updates = Vec::with_capacity(16);

		Self {
			staging_buffer_allocator,
			async_transfers,
			transfer_queue,
			transfer_future: Default::default(),
			buffer_updates,
		}
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
		assert!(data.len() > 0);

		if self.async_transfers.len() == self.async_transfers.capacity() {
			log::warn!(
				"Re-allocating `Vec` for asynchronous transfers to {}! Consider increasing its initial capacity.",
				self.async_transfers.len() + 1
			);
		}
		self.async_transfers.push(Box::new(StagingWork::CopyBuffer { data, dst_buf }));
	}

	/// Add staging work for a new image.
	/// If there is a separate transfer queue, this will be performed asynchronously with the CPU
	/// building the draw command buffers.
	///
	/// If an empty slice is given to `regions`, this will use the default value produced by
	/// Vulkano's `CopyBufferToImageInfo::buffer_image`.
	pub fn copy_to_image<Px>(&mut self, data: Vec<Px>, dst_image: Arc<Image>, regions: &[BufferImageCopy])
	where
		Px: BufferContents + Copy,
	{
		assert!(data.len() > 0);
		assert!(dst_image.format().block_size().is_power_of_two());

		if self.async_transfers.len() == self.async_transfers.capacity() {
			log::warn!(
				"Re-allocating `Vec` for asynchronous transfers to {}! Consider increasing its initial capacity.",
				self.async_transfers.len() + 1
			);
		}
		self.async_transfers.push(Box::new(StagingWork::CopyBufferToImage {
			data,
			dst_image,
			regions: regions.into(),
		}));
	}

	/// Update a buffer at the begninning of the next graphics submission.
	pub fn update_buffer<T>(&mut self, data: Box<[T]>, dst_buf: Subbuffer<[T]>)
	where
		T: BufferContents + Copy,
	{
		assert!(data.len() > 0);

		if self.buffer_updates.len() == self.buffer_updates.capacity() {
			log::warn!(
				"Re-allocating `Vec` for buffer updates to {}! Consider increasing its initial capacity.",
				self.buffer_updates.len() + 1
			);
		}

		self.buffer_updates.push(Box::new(UpdateBufferData { dst_buf, data }));
	}

	/// Submit the asynchronous transfers that are waiting.
	/// Run this just before beginning to build the draw command buffers,
	/// so that the transfers can be done while the CPU is busy with building the draw command buffers.
	///
	/// This does nothing if there is no asynchronous transfer queue. In such a case, the transfers will
	/// instead be done at the beginning of the graphics submission on the graphics queue.
	pub fn submit_async_transfers(&mut self, command_buffer_allocator: &StandardCommandBufferAllocator) -> crate::Result<()>
	{
		let mut staging_buf_alloc_guard = self.staging_buffer_allocator.lock().unwrap();

		let staging_buf_usage_frame = self
			.async_transfers
			.iter()
			.filter(|c| !c.will_use_update_buffer())
			.map(|c| c.device_layout())
			.reduce(|usage_frame, device_layout| usage_frame.extend(device_layout).unwrap().0);

		// gather stats on staging buffer usage
		if let Some(usage) = staging_buf_usage_frame {
			let needed_arena_size = usage.pad_to_alignment().size();
			if needed_arena_size > staging_buf_alloc_guard.arena_size() {
				log::debug!("reserving {needed_arena_size} bytes in `SubbufferAllocator`");
			}
			staging_buf_alloc_guard.reserve(needed_arena_size)?;

			// TODO: If staging buffer usage exceeds a certain threshold, don't allocate any further,
			// and defer any further transfers to a separate submission.
			// We could also instead make it allocate everything it needs at once, then reset the arena size
			// to below the threshold during the next submission.
		}

		if let Some(q) = &self.transfer_queue {
			if self.async_transfers.len() > 0 {
				let mut cb = AutoCommandBufferBuilder::primary(
					command_buffer_allocator,
					q.queue_family_index(),
					CommandBufferUsage::OneTimeSubmit,
				)?;

				for work in self.async_transfers.drain(..) {
					work.add_command(&mut cb, &mut staging_buf_alloc_guard);
				}

				let transfer_future = cb.build()?.execute(q.clone()).unwrap().then_signal_fence_and_flush()?;

				// This panics here if there's an unused future, because it *must* have been used when
				// the draw commands were submitted last frame. Otherwise, we can't guarantee that transfers
				// have finished by the time the draws that need them are performed.
				assert!(self.transfer_future.replace(transfer_future).is_none());
			}
		}

		Ok(())
	}

	pub fn add_synchronous_transfer_commands(&mut self, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
	{
		// buffer updates
		for buf_update in self.buffer_updates.drain(..) {
			buf_update.add_command(cb_builder);
		}

		// do async transfers that couldn't be submitted earlier
		let mut staging_buf_alloc_guard = self.staging_buffer_allocator.lock().unwrap();
		for work in self.async_transfers.drain(..) {
			work.add_command(cb_builder, &mut staging_buf_alloc_guard);
		}
	}

	pub fn take_transfer_future(&mut self) -> Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>
	{
		self.transfer_future.take()
	}
}

/// Synchronous buffer updates that must be submitted on the graphics queue.
struct UpdateBufferData<T: BufferContents + Copy>
{
	dst_buf: Subbuffer<[T]>,
	data: Box<[T]>,
}
impl<T: BufferContents + Copy> UpdateBufferDataTrait for UpdateBufferData<T>
{
	fn add_command(self: Box<Self>, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
	{
		cb_builder.update_buffer(self.dst_buf, self.data).unwrap();
	}
}
trait UpdateBufferDataTrait: Send + Sync
{
	fn add_command(self: Box<Self>, _: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>);
}

/// Asynchronous staging work that can be submitted to the transfer queue, if there is one.
enum StagingWork<T: BufferContents + Copy>
{
	CopyBuffer
	{
		data: Vec<T>, dst_buf: Subbuffer<[T]>
	},
	CopyBufferToImage
	{
		data: Vec<T>,
		dst_image: Arc<Image>,
		regions: Vec<BufferImageCopy>,
	},
}
impl<T: BufferContents + Copy> StagingWorkTrait for StagingWork<T>
{
	fn device_layout(&self) -> DeviceLayout
	{
		let (Self::CopyBuffer { data, .. } | Self::CopyBufferToImage { data, .. }) = self;

		let data_size_bytes: DeviceSize = (data.len() * std::mem::size_of::<T>()).try_into().unwrap();
		let alignment = match self {
			Self::CopyBuffer { .. } => T::LAYOUT.alignment(),
			Self::CopyBufferToImage { dst_image, .. } => dst_image.format().block_size().try_into().unwrap(),
		};
		DeviceLayout::new(data_size_bytes.try_into().unwrap(), alignment).unwrap()
	}

	// If the data is small enough, this will use `update_buffer` instead.
	fn will_use_update_buffer(&self) -> bool
	{
		match self {
			Self::CopyBuffer { data, .. } => {
				let data_size_bytes = std::mem::size_of::<T>() * data.len();
				data_size_bytes <= 65536
			}
			_ => false,
		}
	}

	fn add_command(
		self: Box<Self>,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		subbuffer_allocator: &mut SubbufferAllocator,
	)
	{
		// We allocate a subbuffer using a `DeviceLayout` here so that it's aligned to the block
		// size of the format. We also `unwrap` on `allocate` because it shouldn't allocate a new
		// arena (meaning memory allocation errors shouldn't occur), since we already reserved
		// enough memory for all staging buffers this frame.
		let device_layout = self.device_layout();

		let use_update_buffer = self.will_use_update_buffer();

		match *self {
			Self::CopyBuffer { data, dst_buf } => {
				if use_update_buffer {
					cb_builder.update_buffer(dst_buf, data.into_boxed_slice()).unwrap();
				} else {
					let staging_buf: Subbuffer<[T]> = subbuffer_allocator.allocate(device_layout).unwrap().reinterpret();
					staging_buf.write().unwrap().copy_from_slice(&data);
					cb_builder.copy_buffer(CopyBufferInfo::buffers(staging_buf, dst_buf)).unwrap();
				}
			}
			Self::CopyBufferToImage {
				data,
				dst_image,
				regions,
				..
			} => {
				let staging_buf: Subbuffer<[T]> = subbuffer_allocator.allocate(device_layout).unwrap().reinterpret();
				staging_buf.write().unwrap().copy_from_slice(&data);
				let with_default_region = CopyBufferToImageInfo::buffer_image(staging_buf, dst_image);
				let copy_info = if regions.len() > 0 {
					CopyBufferToImageInfo {
						regions: regions.clone().into(),
						..with_default_region
					}
				} else {
					with_default_region
				};
				cb_builder.copy_buffer_to_image(copy_info).unwrap();
			}
		}
	}
}
trait StagingWorkTrait: Send + Sync
{
	fn device_layout(&self) -> DeviceLayout;

	fn will_use_update_buffer(&self) -> bool;

	fn add_command(self: Box<Self>, _: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, _: &mut SubbufferAllocator);
}
