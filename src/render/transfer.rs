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
	allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage,
	CopyBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::memory::allocator::{DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::{
	future::{FenceSignalFuture, NowFuture},
	GpuFuture,
};
use vulkano::DeviceSize;

pub struct TransferManager
{
	// Transfers to initialize buffers and images. If there is an asynchronous transfer queue,
	// these will be performed while the CPU is busy with building the draw command buffers.
	// Otherwise, these will be run at the beginning of the next graphics submission, before draws are performed.
	async_transfers: Vec<StagingWork>,
	transfer_queue: Option<Arc<Queue>>, // if there is a separate (preferably dedicated) transfer queue, use it for transfers
	transfer_future: Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>,

	// Buffer updates to run at the beginning of the next graphics submission.
	buffer_updates: Vec<Box<dyn UpdateBufferDataTrait>>,
	staging_buffer_allocator: Mutex<SubbufferAllocator>,
	staging_buf_max_size: DeviceSize, // Maximum staging buffer usage for the entire duration of the program.
	staging_buf_usage_frame: DeviceSize,
}
impl TransferManager
{
	pub fn new(transfer_queue: Option<Arc<Queue>>, memory_allocator: Arc<StandardMemoryAllocator>) -> Self
	{
		let pool_create_info = SubbufferAllocatorCreateInfo {
			arena_size: 8 * 1024 * 1024, // this should be adjusted based on actual memory usage
			buffer_usage: BufferUsage::TRANSFER_SRC,
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buffer_allocator = Mutex::new(SubbufferAllocator::new(memory_allocator.clone(), pool_create_info));

		// the capacity of these should be adjusted based on number of transfers that might be done in one frame
		let buffer_updates = Vec::with_capacity(16);
		let async_transfers = Vec::with_capacity(200);

		Self {
			async_transfers,
			transfer_queue,
			transfer_future: Default::default(),
			buffer_updates,
			staging_buffer_allocator,
			staging_buf_max_size: 0,
			staging_buf_usage_frame: 0,
		}
	}

	/// Add staging work for new objects.
	/// If there is a separate transfer queue, this will be performed asynchronously with the CPU
	/// building the draw command buffers.
	pub fn add_transfer(&mut self, work: StagingWork)
	{
		self.staging_buf_usage_frame += work.buf_size();
		if self.async_transfers.len() == self.async_transfers.capacity() {
			log::warn!(
				"Re-allocating `Vec` for asynchronous transfers to {}! Consider increasing its initial capacity.",
				self.async_transfers.len() + 1
			);
		}
		self.async_transfers.push(work);
	}

	/// Update a buffer at the begninning of the next graphics submission.
	pub fn update_buffer<T>(&mut self, data: &[T], dst_buf: Subbuffer<[T]>)
	where
		T: BufferContents + Copy,
	{
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
		if let Some(q) = self.transfer_queue.as_ref() {
			if self.async_transfers.len() > 0 {
				let mut cb = AutoCommandBufferBuilder::primary(
					command_buffer_allocator,
					q.queue_family_index(),
					CommandBufferUsage::OneTimeSubmit,
				)?;

				for work in self.async_transfers.drain(..) {
					work.add_command(&mut cb);
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
		// buffer updates
		if self.buffer_updates.len() > 0 {
			let mut staging_buf_alloc_guard = self.staging_buffer_allocator.lock().unwrap();

			for buf_update in self.buffer_updates.drain(..) {
				buf_update.add_command(cb_builder, &mut staging_buf_alloc_guard)?;
				self.staging_buf_usage_frame += buf_update.data_size();
			}

			// gather stats on staging buffer usage
			if self.staging_buf_usage_frame > self.staging_buf_max_size {
				self.staging_buf_max_size = self.staging_buf_usage_frame;
				log::debug!("max staging buffer usage per frame: {} bytes", self.staging_buf_max_size);
			}

			self.staging_buf_usage_frame = 0;
		}

		// do async transfers that couldn't be submitted earlier
		for work in self.async_transfers.drain(..) {
			work.add_command(cb_builder);
		}

		Ok(())
	}

	pub fn get_staging_buffer<T>(&self, len: DeviceSize) -> crate::Result<Subbuffer<[T]>>
	where
		T: BufferContents + Copy,
	{
		let staging_buf = self.staging_buffer_allocator.lock().unwrap().allocate_slice(len)?;
		Ok(staging_buf)
	}

	pub fn get_tex_staging_buffer<Px>(&self, data: &[Px], format: Format) -> crate::Result<Subbuffer<[Px]>>
	where
		Px: BufferContents + Copy,
	{
		// We allocate a subbuffer using a `DeviceLayout` here so that it's aligned to the block size
		// of the format.
		let data_size_bytes = (data.len() * std::mem::size_of::<Px>()).try_into().unwrap();
		let device_layout = DeviceLayout::from_size_alignment(data_size_bytes, format.block_size())
			.ok_or("Texture::new_from_slice: slice is empty or alignment is not a power of two")?;

		let staging_buf: Subbuffer<[Px]> = self
			.staging_buffer_allocator
			.lock()
			.unwrap()
			.allocate(device_layout)?
			.reinterpret();

		staging_buf.write().unwrap().copy_from_slice(data);

		Ok(staging_buf)
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
	fn data_size(&self) -> DeviceSize
	{
		self.dst_buf.size()
	}

	fn add_command(
		&self,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		subbuffer_allocator: &mut SubbufferAllocator,
	) -> crate::Result<()>
	{
		let staging_buf = subbuffer_allocator.allocate_slice(self.data.len().try_into().unwrap())?;
		staging_buf.write().unwrap().copy_from_slice(self.data.as_slice());

		// TODO: actually use `update_buffer` when the `'static` requirement gets removed for the data
		cb_builder.copy_buffer(CopyBufferInfo::buffers(staging_buf, self.dst_buf.clone()))?;

		Ok(())
	}
}
trait UpdateBufferDataTrait: Send + Sync
{
	fn data_size(&self) -> DeviceSize;

	fn add_command(
		&self,
		_: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		_: &mut SubbufferAllocator,
	) -> crate::Result<()>;
}

pub enum StagingWork
{
	CopyBuffer(CopyBufferInfo),
	CopyBufferToImage(CopyBufferToImageInfo),
}
impl StagingWork
{
	fn add_command(self, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
	{
		match self {
			StagingWork::CopyBuffer(info) => cb_builder.copy_buffer(info).unwrap(),
			StagingWork::CopyBufferToImage(info) => cb_builder.copy_buffer_to_image(info).unwrap(),
		};
	}

	fn buf_size(&self) -> DeviceSize
	{
		match self {
			StagingWork::CopyBuffer(info) => info.src_buffer.size(),
			StagingWork::CopyBufferToImage(info) => info.src_buffer.size(),
		}
	}
}
impl From<CopyBufferInfo> for StagingWork
{
	fn from(info: CopyBufferInfo) -> StagingWork
	{
		Self::CopyBuffer(info)
	}
}
impl From<CopyBufferToImageInfo> for StagingWork
{
	fn from(info: CopyBufferToImageInfo) -> StagingWork
	{
		Self::CopyBufferToImage(info)
	}
}
