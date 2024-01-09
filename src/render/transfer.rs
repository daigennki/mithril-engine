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

	/// Add staging work for a new buffer. (use `update_buffer` instead for previously created buffers)
	/// If there is a separate transfer queue, this will be performed asynchronously with the CPU
	/// building the draw command buffers.
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
	pub fn update_buffer<T>(&mut self, data: &[T], dst_buf: Subbuffer<[T]>)
	where
		T: BufferContents + Copy,
	{
		assert!(data.len() > 0);

		// This will be submitted to the graphics queue since we're copying to an existing buffer,
		// which might be in use by a previous submission.
		if self.buffer_updates.len() == self.buffer_updates.capacity() {
			log::warn!(
				"Re-allocating `Vec` for buffer updates to {}! Consider increasing its initial capacity.",
				self.buffer_updates.len() + 1
			);
		}

		self.buffer_updates.push(Box::new(UpdateBufferData {
			dst_buf,
			data: data.into(),
		}));
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

		let copy_layouts = self.async_transfers.iter().map(|c| c.device_layout());
		let update_layouts = self.buffer_updates.iter().map(|u| u.device_layout());
		let staging_buf_usage_frame = copy_layouts
			.chain(update_layouts)
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
					work.add_command(&mut cb, &mut staging_buf_alloc_guard)?;
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

	pub fn add_synchronous_transfer_commands(
		&mut self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
	) -> crate::Result<()>
	{
		let mut staging_buf_alloc_guard = self.staging_buffer_allocator.lock().unwrap();

		// buffer updates
		for buf_update in self.buffer_updates.drain(..) {
			buf_update.add_command(cb_builder, &mut staging_buf_alloc_guard)?;
		}

		// do async transfers that couldn't be submitted earlier
		for work in self.async_transfers.drain(..) {
			work.add_command(cb_builder, &mut staging_buf_alloc_guard)?;
		}

		Ok(())
	}

	pub fn take_transfer_future(&mut self) -> Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>
	{
		self.transfer_future.take()
	}
}

struct UpdateBufferData<T: BufferContents + Copy>
{
	dst_buf: Subbuffer<[T]>,
	data: Vec<T>,
}
impl<T: BufferContents + Copy> UpdateBufferDataTrait for UpdateBufferData<T>
{
	fn device_layout(&self) -> DeviceLayout
	{
		let data_size_bytes: DeviceSize = (self.data.len() * std::mem::size_of::<T>()).try_into().unwrap();
		DeviceLayout::new(data_size_bytes.try_into().unwrap(), T::LAYOUT.alignment()).unwrap()
	}

	fn add_command(
		self: Box<Self>,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		subbuffer_allocator: &mut SubbufferAllocator,
	) -> crate::Result<()>
	{
		let staging_buf = subbuffer_allocator.allocate_slice(self.data.len().try_into().unwrap())?;
		staging_buf.write().unwrap().copy_from_slice(&self.data);

		// TODO: actually use `update_buffer` when the `'static` requirement gets removed for the data
		cb_builder.copy_buffer(CopyBufferInfo::buffers(staging_buf, self.dst_buf))?;

		Ok(())
	}
}
trait UpdateBufferDataTrait: Send + Sync
{
	fn device_layout(&self) -> DeviceLayout;

	fn add_command(
		self: Box<Self>,
		_: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		_: &mut SubbufferAllocator,
	) -> crate::Result<()>;
}

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
impl<T: BufferContents + Copy> StagingWork<T>
{
	fn data(&self) -> &Vec<T>
	{
		let (Self::CopyBuffer { data, .. } | Self::CopyBufferToImage { data, .. }) = self;
		data
	}
}
impl<T: BufferContents + Copy> StagingWorkTrait for StagingWork<T>
{
	fn device_layout(&self) -> DeviceLayout
	{
		let data_size_bytes: DeviceSize = (self.data().len() * std::mem::size_of::<T>()).try_into().unwrap();
		let alignment = match self {
			Self::CopyBuffer { .. } => T::LAYOUT.alignment(),
			Self::CopyBufferToImage { dst_image, .. } => dst_image.format().block_size().try_into().unwrap(),
		};
		DeviceLayout::new(data_size_bytes.try_into().unwrap(), alignment).unwrap()
	}

	fn add_command(
		self: Box<Self>,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		subbuffer_allocator: &mut SubbufferAllocator,
	) -> crate::Result<()>
	{
		// We allocate a subbuffer using a `DeviceLayout` here so that it's aligned to the block size
		// of the format.
		let device_layout = self.device_layout();
		let staging_buf: Subbuffer<[T]> = subbuffer_allocator.allocate(device_layout)?.reinterpret();
		staging_buf.write().unwrap().copy_from_slice(self.data());

		match *self {
			Self::CopyBuffer { dst_buf, .. } => {
				cb_builder.copy_buffer(CopyBufferInfo::buffers(staging_buf, dst_buf))?;
			},
			Self::CopyBufferToImage { dst_image, regions, .. } => {
				let with_default_region = CopyBufferToImageInfo::buffer_image(staging_buf, dst_image);
				let copy_info = if regions.len() > 0 {
					CopyBufferToImageInfo {
						regions: regions.clone().into(),
						..with_default_region
					}
				} else {
					with_default_region
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

	fn add_command(
		self: Box<Self>,
		_: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		_: &mut SubbufferAllocator,
	) -> crate::Result<()>;
}
