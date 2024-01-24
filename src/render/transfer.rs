/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::sync::Arc;

use vulkano::buffer::{
	allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
	BufferContents, BufferUsage, Subbuffer,
};
use vulkano::command_buffer::{
	allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BufferImageCopy, CommandBufferExecFuture,
	CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
};
use vulkano::device::{Device, Queue};
use vulkano::image::{Image, ImageSubresourceLayers};
use vulkano::memory::{
	allocator::{BumpAllocator, DeviceLayout, GenericMemoryAllocator, GenericMemoryAllocatorCreateInfo, MemoryTypeFilter},
	DeviceAlignment, MemoryProperties, MemoryPropertyFlags,
};
use vulkano::sync::{
	future::{FenceSignalFuture, NowFuture},
	GpuFuture,
};
use vulkano::DeviceSize;

pub struct TransferManager
{
	bump_allocator: Arc<GenericMemoryAllocator<BumpAllocator>>,

	// Transfers to initialize buffers and images. If there is an asynchronous transfer queue,
	// these will be performed while the CPU is busy with building the draw command buffers.
	// Otherwise, these will be run at the beginning of the next graphics submission, before draws
	// are performed.
	async_transfers: Option<Box<dyn StagingWorkTrait>>,
	transfer_queue: Option<Arc<Queue>>,
	transfer_future: Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>,

	next_arena_size: Option<DeviceSize>,

	// Buffer updates to run at the beginning of the next graphics submission.
	buffer_updates: Option<Box<dyn UpdateBufferDataTrait>>,
}
impl TransferManager
{
	pub fn new(device: Arc<Device>, transfer_queue: Option<Arc<Queue>>) -> Self
	{
		let MemoryProperties {
			memory_types,
			memory_heaps,
			..
		} = device.physical_device().memory_properties();

		let mut block_sizes = vec![0; memory_types.len()];
		let mut memory_type_bits = u32::MAX;

		for (index, memory_type) in memory_types.iter().enumerate() {
			const LARGE_HEAP_THRESHOLD: DeviceSize = 1024 * 1024 * 1024;

			let heap_size = memory_heaps[memory_type.heap_index as usize].size;

			block_sizes[index] = if heap_size >= LARGE_HEAP_THRESHOLD {
				256 * 1024 * 1024
			} else {
				64 * 1024 * 1024
			};

			if memory_type.property_flags.intersects(
				MemoryPropertyFlags::LAZILY_ALLOCATED
					| MemoryPropertyFlags::PROTECTED
					| MemoryPropertyFlags::DEVICE_COHERENT
					| MemoryPropertyFlags::RDMA_CAPABLE,
			) {
				// VUID-VkMemoryAllocateInfo-memoryTypeIndex-01872
				// VUID-vkAllocateMemory-deviceCoherentMemory-02790
				// Lazily allocated memory would just cause problems for suballocation in general.
				memory_type_bits &= !(1 << index);
			}
		}

		let allocator_create_info = GenericMemoryAllocatorCreateInfo {
			block_sizes: &block_sizes,
			memory_type_bits,
			..Default::default()
		};
		let bump_allocator = Arc::new(GenericMemoryAllocator::<BumpAllocator>::new(device, allocator_create_info));

		Self {
			bump_allocator,
			async_transfers: None,
			transfer_queue,
			transfer_future: Default::default(),
			next_arena_size: None,
			buffer_updates: None,
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
		assert!(!data.is_empty());

		let work = StagingWork {
			data,
			dst: StagingDst::Buffer(dst_buf),
			next: self.async_transfers.take(),
		};
		self.async_transfers = Some(Box::new(work));
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
			next: self.async_transfers.take(),
		};
		self.async_transfers = Some(Box::new(work));
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

	/// Submit the asynchronous transfers that are waiting.
	/// Run this just before beginning to build the draw command buffers,
	/// so that the transfers can be done while the CPU is busy with building the draw command buffers.
	///
	/// This does nothing if there is no asynchronous transfer queue. In such a case, the transfers will
	/// instead be done at the beginning of the graphics submission on the graphics queue.
	pub fn submit_async_transfers(&mut self, command_buffer_allocator: &StandardCommandBufferAllocator) -> crate::Result<()>
	{
		// calculate how large the total staging buffer usage will be
		let mut async_transfer_option = self.async_transfers.as_ref().map(|inner_box| inner_box.as_ref());
		let mut staging_buf_usage_frame: Option<DeviceLayout> = None;
		while let Some(work) = async_transfer_option.take() {
			if !work.will_use_update_buffer() {
				let device_layout = work.device_layout();
				staging_buf_usage_frame = staging_buf_usage_frame
					.take()
					.map(|usage_frame| usage_frame.extend(device_layout).unwrap().0)
					.or(Some(device_layout));
			}
			async_transfer_option = work.next_ref();
		}

		if let Some(usage) = staging_buf_usage_frame {
			self.next_arena_size = Some(usage.pad_to_alignment().size());
		}

		if let Some(q) = self.transfer_queue.clone() {
			if self.async_transfers.is_some() {
				let mut cb = AutoCommandBufferBuilder::primary(
					command_buffer_allocator,
					q.queue_family_index(),
					CommandBufferUsage::OneTimeSubmit,
				)?;

				self.add_copies(&mut cb);

				let transfer_future = cb.build()?.execute(q).unwrap().then_signal_fence_and_flush()?;

				// This panics here if there's an unused future, because it *must* have been used when
				// the draw commands were submitted last frame. Otherwise, we can't guarantee that transfers
				// have finished by the time the draws that need them are performed.
				assert!(self.transfer_future.replace(transfer_future).is_none());
			}
		}

		Ok(())
	}

	pub fn add_synchronous_transfer_commands(&mut self, cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
	{
		// buffer updates
		let mut buf_update_option = self.buffer_updates.take();
		while let Some(buf_update) = buf_update_option.take() {
			buf_update_option = buf_update.add_command(cb);
		}

		// do async transfers that couldn't be submitted earlier
		self.add_copies(cb);
	}

	fn add_copies(&mut self, cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>)
	{
		if let Some(arena_size) = self.next_arena_size.take() {
			let pool_create_info = SubbufferAllocatorCreateInfo {
				arena_size,
				buffer_usage: BufferUsage::TRANSFER_SRC,
				memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			};
			let mut staging_buf_alloc = SubbufferAllocator::new(self.bump_allocator.clone(), pool_create_info);

			let mut staging_work_option = self.async_transfers.take();
			while let Some(work) = staging_work_option.take() {
				staging_work_option = work.add_command(cb, &mut staging_buf_alloc);
			}
		}
	}

	pub fn take_transfer_future(&mut self) -> Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>>
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

/// Asynchronous staging work that can be submitted to the transfer queue, if there is one. This
/// is a singly linked list, and the next staging work is returned by `add_command`.
///
/// Note for when this is used with images:
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
	next: Option<Box<dyn StagingWorkTrait>>,
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

	/// Add the command for this staging work, and return the next work (if there is any).
	fn add_command(
		self: Box<Self>,
		cb_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		subbuffer_allocator: &mut SubbufferAllocator<GenericMemoryAllocator<BumpAllocator>>,
	) -> Option<Box<dyn StagingWorkTrait>>
	{
		// We allocate a subbuffer using a `DeviceLayout` here so that it's aligned to the block
		// size of the format. We also `unwrap` on `allocate` because it shouldn't allocate a new
		// arena (meaning memory allocation errors shouldn't occur), since we already reserved
		// enough memory for all staging buffers this frame.
		let device_layout = self.device_layout();

		let use_update_buffer = self.will_use_update_buffer();

		match self.dst {
			StagingDst::Buffer(dst_buf) => {
				if use_update_buffer {
					cb_builder.update_buffer(dst_buf, self.data.into_boxed_slice()).unwrap();
				} else {
					let staging_buf: Subbuffer<[T]> = subbuffer_allocator.allocate(device_layout).unwrap().reinterpret();
					staging_buf.write().unwrap().copy_from_slice(&self.data);
					cb_builder.copy_buffer(CopyBufferInfo::buffers(staging_buf, dst_buf)).unwrap();
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

					buffer_offset += super::get_mip_size(format, mip_width, mip_height);
					mip_width /= 2;
					mip_height /= 2;
				}

				let staging_buf: Subbuffer<[T]> = subbuffer_allocator.allocate(device_layout).unwrap().reinterpret();
				staging_buf.write().unwrap().copy_from_slice(&self.data);

				let copy_info = CopyBufferToImageInfo {
					regions: regions.into(),
					..CopyBufferToImageInfo::buffer_image(staging_buf, dst_image)
				};
				cb_builder.copy_buffer_to_image(copy_info).unwrap();
			}
		}

		self.next
	}

	fn next_ref(&self) -> Option<&dyn StagingWorkTrait>
	{
		self.next.as_ref().map(|next_inner| next_inner.as_ref())
	}
}
trait StagingWorkTrait: Send + Sync
{
	fn device_layout(&self) -> DeviceLayout;

	fn will_use_update_buffer(&self) -> bool;

	fn add_command(
		self: Box<Self>,
		_: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		_: &mut SubbufferAllocator<GenericMemoryAllocator<BumpAllocator>>,
	) -> Option<Box<dyn StagingWorkTrait>>;

	fn next_ref(&self) -> Option<&dyn StagingWorkTrait>;
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
