/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
	allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
	AutoCommandBufferBuilder, BufferImageCopy, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo,
	PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
};
use vulkano::device::{DeviceOwned, Queue};
use vulkano::image::{Image, ImageSubresourceLayers};
use vulkano::memory::{
	allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator},
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
	staging_arenas: [Subbuffer<[u8]>; 2],
	current_arena: usize,
	staging_layout: Option<DeviceLayout>, // Current layout inside the current arena.

	queue: Arc<Queue>, // The queue to submit the transfers to.
	transfer_future: Option<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,
}
impl TransferManager
{
	/// Create a transfer manager that will submit transfers to the given `queue`. It'll create a
	/// big buffer which will be split up into subbuffers for use as staging buffers.
	pub fn new(queue: Arc<Queue>, memory_allocator: Arc<StandardMemoryAllocator>) -> crate::Result<Self>
	{
		let device = queue.device().clone();
		let cb_alloc_info = StandardCommandBufferAllocatorCreateInfo {
			primary_buffer_count: 4,
			..Default::default()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(device, cb_alloc_info);

		let buf_info = BufferCreateInfo {
			usage: BufferUsage::TRANSFER_SRC,
			..Default::default()
		};
		let alloc_info = AllocationCreateInfo {
			memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
			..Default::default()
		};
		let total_arena_size: DeviceSize = STAGING_ARENA_SIZE * 2;
		let combined_staging_arenas = Buffer::new_slice(memory_allocator, buf_info, alloc_info, total_arena_size)?;

		Ok(Self {
			command_buffer_allocator,
			transfers: Vec::with_capacity(256),
			staging_arenas: combined_staging_arenas.split_at(STAGING_ARENA_SIZE).into(),
			current_arena: 0,
			staging_layout: None,
			queue,
			transfer_future: Default::default(),
		})
	}

	fn add_transfer<T>(&mut self, data: &[T], dst: StagingDst) -> crate::Result<()>
	where
		T: BufferContents + Copy,
	{
		// We get a subbuffer using a `DeviceLayout` here so that it's aligned to the block size of
		// the format.
		let data_size_bytes: DeviceSize = std::mem::size_of_val(data).try_into().unwrap();
		let nonzero_size = data_size_bytes.try_into().expect("`data` for transfer is empty");
		let transfer_layout = DeviceLayout::new(nonzero_size, dst.alignment()).unwrap();

		if !self.transfers.is_empty() {
			let (extended_usage, _) = self.staging_layout.unwrap().extend(transfer_layout).unwrap();
			if extended_usage.size() > STAGING_ARENA_SIZE || self.transfers.len() == self.transfers.capacity() {
				log::debug!(
					"staging buffer arena size or transfer `Vec` capacity reached, submitting pending transfers now..."
				);
				self.submit_transfers()?;
			}
		}

		let (new_layout, new_offset) = if let Some(current_layout) = self.staging_layout.take() {
			current_layout.extend(transfer_layout).unwrap()
		} else {
			(transfer_layout, 0)
		};
		self.staging_layout = Some(new_layout);

		let slice_end = new_offset + transfer_layout.size();
		if slice_end > self.staging_arenas[self.current_arena].size() {
			return Err("Transfer too big for staging buffer arena!".into());
		}
		let staging_buf: Subbuffer<[T]> = self.staging_arenas[self.current_arena]
			.clone()
			.slice(new_offset..slice_end)
			.reinterpret();
		staging_buf.write().unwrap().copy_from_slice(data);

		let work = StagingWork {
			src: staging_buf.into_bytes(),
			dst,
		};
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

			self.staging_layout = None;
			self.current_arena += 1;
			if self.current_arena == self.staging_arenas.len() {
				self.current_arena = 0;
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
	fn add_command(self, cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
	{
		match self.dst {
			StagingDst::Buffer(dst_buf) => cb_builder.copy_buffer(CopyBufferInfo::buffers(self.src, dst_buf)),
			StagingDst::Image(dst_image) => {
				let format = dst_image.format();
				let mip_levels = dst_image.mip_levels();
				let array_layers = dst_image.array_layers();

				// generate copies for every mipmap level
				let mut regions = smallvec::SmallVec::with_capacity(mip_levels as usize);
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
					regions,
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
