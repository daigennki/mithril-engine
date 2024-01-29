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
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	AutoCommandBufferBuilder, BufferImageCopy, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo,
	PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
};
use vulkano::device::{DeviceOwned, Queue};
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
	command_buffer_allocator: StandardCommandBufferAllocator,

	// Transfers to initialize buffers and images.
	// If this or the staging buffer arena gets full, the transfers will get submitted immediately.
	transfers: Vec<StagingWork>,
	staging_buffer_allocator: Mutex<SubbufferAllocator>,
	queue: Arc<Queue>, // The queue to submit the transfers to.
	transfer_future: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,

	// Buffer updates to run at the beginning of the next graphics presentation submission.
	buffer_updates: Vec<UpdateBufferData>,
}
impl TransferManager
{
	/// Create a transfer manager that will submit transfers to the given `queue`, and create a
	/// `SubbufferAllocator` using the given `memory_allocator`.
	pub fn new(queue: Arc<Queue>, memory_allocator: Arc<StandardMemoryAllocator>) -> Self
	{
		let pool_create_info = SubbufferAllocatorCreateInfo {
			arena_size: STAGING_ARENA_SIZE,
			buffer_usage: BufferUsage::TRANSFER_SRC,
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let staging_buffer_allocator = Mutex::new(SubbufferAllocator::new(memory_allocator.clone(), pool_create_info));

		let device = queue.device().clone();
		let cb_alloc_info = StandardCommandBufferAllocatorCreateInfo {
			primary_buffer_count: 4,
			..Default::default()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(device, cb_alloc_info);

		Self {
			command_buffer_allocator,
			transfers: Vec::with_capacity(256),
			staging_buffer_allocator,
			queue,
			transfer_future: Default::default(),
			buffer_updates: Vec::new(),
		}
	}

	fn add_transfer<T>(&mut self, data: &[T], dst: StagingDst) -> crate::Result<()>
	where
		T: BufferContents + Copy,
	{
		assert!(!data.is_empty());

		// We allocate a subbuffer using a `DeviceLayout` here so that it's aligned to the block
		// size of the format.
		let data_size_bytes: DeviceSize = std::mem::size_of_val(data).try_into().unwrap();
		let transfer_layout = DeviceLayout::new(data_size_bytes.try_into().unwrap(), dst.alignment()).unwrap();
		let staging_buf: Subbuffer<[T]> = {
			let staging_buf_alloc_guard = self.staging_buffer_allocator.lock().unwrap();
			staging_buf_alloc_guard.allocate(transfer_layout)?.reinterpret()
		};
		staging_buf.write().unwrap().copy_from_slice(data);

		let work = StagingWork {
			src: staging_buf.into_bytes(),
			dst,
		};

		// Calculate the total staging buffer usage for the pending transfers.
		let will_exceed_arena_size = self
			.transfers
			.iter()
			.map(|pending_work| pending_work.device_layout())
			.reduce(|acc, layout| acc.extend(layout).unwrap().0)
			.map(|pending_transfer_size| {
				let (extended_layout, _) = pending_transfer_size.extend(transfer_layout).unwrap();
				extended_layout.size() > STAGING_ARENA_SIZE
			})
			.unwrap_or(false);

		// If adding this transfer would cause staging buffer usage to exceed the staging buffer
		// arena size, or the length of `transfers` to exceed its capacity, submit pending
		// transfers immediately before adding this transfer.
		if will_exceed_arena_size || self.transfers.len() == self.transfers.capacity() {
			log::debug!("staging buffer arena size or transfer `Vec` capacity reached, submitting pending transfers now...");
			self.submit_transfers()?;
		}

		self.transfers.push(work);

		Ok(())
	}

	/// Add staging work for a new buffer.
	///
	/// If the buffer might be in use by a previous submission, use `update_buffer` instead.
	pub fn copy_to_buffer<T>(&mut self, data: &[T], dst_buf: Subbuffer<[T]>) -> crate::Result<()>
	where
		T: BufferContents + Copy,
	{
		self.add_transfer(data, StagingDst::Buffer((dst_buf.into_bytes(), T::LAYOUT.alignment())))
	}

	/// Add staging work for a new image.
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
	pub fn copy_to_image<Px>(&mut self, data: &[Px], dst_image: Arc<Image>) -> crate::Result<()>
	where
		Px: BufferContents + Copy,
	{
		assert!(dst_image.format().block_size().is_power_of_two());

		self.add_transfer(data, StagingDst::Image(dst_image))
	}

	/// Update a buffer at the begninning of the next graphics presentation submission.
	pub fn update_buffer<T>(&mut self, data: &[T], dst: Subbuffer<[T]>)
	where
		T: BufferContents + Copy + bytemuck::Pod,
	{
		let buf_update = UpdateBufferData {
			dst: dst.into_bytes(),
			data: bytemuck::cast_slice::<_, u8>(data).into(),
		};

		if self.buffer_updates.len() == self.buffer_updates.capacity() {
			let old_capacity = self.buffer_updates.capacity();
			self.buffer_updates.reserve(64);
			let new_capacity = self.buffer_updates.capacity();
			log::debug!("increased buffer updates `Vec` capacity from {old_capacity} to {new_capacity}");
		}

		self.buffer_updates.push(buf_update);
	}

	/// Submit pending transfers. Run this just before beginning to build the draw command buffers
	/// so that the transfers can be done while the CPU is busy with building the draw command
	/// buffers.
	pub fn submit_transfers(&mut self) -> crate::Result<()>
	{
		if !self.transfers.is_empty() {
			let mut cb = AutoCommandBufferBuilder::primary(
				&self.command_buffer_allocator,
				self.queue.queue_family_index(),
				CommandBufferUsage::OneTimeSubmit,
			)?;

			for work in self.transfers.drain(..) {
				work.add_command(&mut cb)?;
			}

			let transfer_future = if let Some(f) = self.transfer_future.take() {
				// wait so that we don't allocate too many staging buffers
				f.wait(None)?;

				cb.build()?
					.execute_after(f, self.queue.clone())
					.unwrap()
					.boxed_send_sync()
					.then_signal_fence_and_flush()?
			} else {
				cb.build()?
					.execute(self.queue.clone())
					.unwrap()
					.boxed_send_sync()
					.then_signal_fence_and_flush()?
			};

			self.transfer_future = Some(transfer_future);
		}

		Ok(())
	}

	pub fn add_update_commands(&mut self, cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
	{
		for buf_update in self.buffer_updates.drain(..) {
			buf_update.add_command(cb);
		}
	}

	pub fn take_transfer_future(&mut self) -> Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>
	{
		self.transfer_future.take()
	}
}

/// Synchronous buffer updates that must be submitted after previous graphics submissions have
/// completed.
struct UpdateBufferData
{
	data: Box<[u8]>,
	dst: Subbuffer<[u8]>,
}
impl UpdateBufferData
{
	fn add_command(self, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
	{
		cb_builder.update_buffer(self.dst, self.data).unwrap();
	}
}

struct StagingWork
{
	src: Subbuffer<[u8]>,
	dst: StagingDst,
}
impl StagingWork
{
	fn device_layout(&self) -> DeviceLayout
	{
		DeviceLayout::new(self.src.size().try_into().unwrap(), self.dst.alignment()).unwrap()
	}

	fn add_command(self, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> crate::Result<()>
	{
		match self.dst {
			StagingDst::Buffer((dst_buf, _)) => {
				cb_builder.copy_buffer(CopyBufferInfo::buffers(self.src, dst_buf))?;
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

				let copy_info = CopyBufferToImageInfo {
					regions: regions.into(),
					..CopyBufferToImageInfo::buffer_image(self.src, dst_image)
				};
				cb_builder.copy_buffer_to_image(copy_info)?;
			}
		}

		Ok(())
	}
}

enum StagingDst
{
	Buffer((Subbuffer<[u8]>, DeviceAlignment)),
	Image(Arc<Image>),
}
impl StagingDst
{
	fn alignment(&self) -> DeviceAlignment
	{
		match self {
			Self::Buffer((_, alignment)) => *alignment,
			Self::Image(image) => image.format().block_size().try_into().unwrap(),
		}
	}
}
