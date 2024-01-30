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
		}
	}

	fn add_transfer<T>(&mut self, data: &[T], dst: StagingDst) -> crate::Result<()>
	where
		T: BufferContents + Copy,
	{
		// We allocate a subbuffer using a `DeviceLayout` here so that it's aligned to the block
		// size of the format.
		let data_size_bytes: DeviceSize = std::mem::size_of_val(data).try_into().unwrap();
		let nonzero_size = data_size_bytes.try_into().expect("`data` for transfer is empty");
		let transfer_layout = DeviceLayout::new(nonzero_size, dst.alignment()).unwrap();

		let staging_buf: Subbuffer<[T]> = {
			let staging_buf_alloc_guard = self.staging_buffer_allocator.lock().unwrap();
			staging_buf_alloc_guard.allocate(transfer_layout)?.reinterpret()
		};
		staging_buf.write().unwrap().copy_from_slice(data);

		let work = StagingWork {
			src: staging_buf.into_bytes(),
			dst,
		};
		self.transfers.push(work);

		let total_usage = self
			.transfers
			.iter()
			.map(|pending_work| pending_work.device_layout())
			.reduce(|acc, layout| acc.extend(layout).unwrap().0)
			.unwrap();
		if total_usage.size() >= STAGING_ARENA_SIZE || self.transfers.len() == self.transfers.capacity() {
			log::debug!("staging buffer arena size or transfer `Vec` capacity reached, submitting pending transfers now...");
			self.submit_transfers()?;
		}

		Ok(())
	}

	/// Add staging work for a new buffer.
	///
	/// If the buffer might be in use by a previous submission, use `update_buffer` instead.
	pub fn copy_to_buffer<T>(&mut self, data: &[T], dst_buf: Subbuffer<[T]>) -> crate::Result<()>
	where
		T: BufferContents + Copy,
	{
		self.add_transfer(data, StagingDst::Buffer(dst_buf.into_bytes()))
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

			self.transfers.drain(..).for_each(|work| work.add_command(&mut cb));

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

	pub fn take_transfer_future(&mut self) -> Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>
	{
		self.transfer_future.take()
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

	fn add_command(self, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
	{
		match self.dst {
			StagingDst::Buffer(dst_buf) => cb_builder.copy_buffer(CopyBufferInfo::buffers(self.src, dst_buf)),
			StagingDst::Image(dst_image) => {
				let format = dst_image.format();
				let mip_levels = dst_image.mip_levels();
				let array_layers = dst_image.array_layers();

				// generate copies for every mipmap level
				let mut regions = Vec::with_capacity(mip_levels as usize);
				let [mut mip_width, mut mip_height, _] = dst_image.extent();
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
				cb_builder.copy_buffer_to_image(copy_info)
			}
		}
		.unwrap();
	}
}

enum StagingDst
{
	Buffer(Subbuffer<[u8]>),
	Image(Arc<Image>),
}
impl StagingDst
{
	fn alignment(&self) -> DeviceAlignment
	{
		match self {
			Self::Buffer(buf) => buf.buffer().memory_requirements().layout.alignment(),
			Self::Image(image) => image.format().block_size().try_into().unwrap(),
		}
	}
}
