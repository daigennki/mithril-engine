/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use std::error::Error;
use vulkano::buffer::AllocateBufferError;
use vulkano::image::AllocateImageError;
use vulkano::memory::allocator::MemoryAllocatorError;
use vulkano::{Validated, ValidationError, VulkanError};

#[derive(Debug)]
pub struct EngineError
{
	source: Option<Box<dyn Error + Send + Sync + 'static>>,
	context: &'static str,
}
impl EngineError
{
	pub fn new<E>(context: &'static str, error: E) -> Self
	where
		E: Error + Send + Sync + 'static,
	{
		Self {
			source: Some(Box::new(error)),
			context,
		}
	}
}
impl std::fmt::Display for EngineError
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		match &self.source {
			Some(e) => write!(f, "{}: {}", self.context, e),
			None => write!(f, "{}", self.context),
		}
	}
}
impl Error for EngineError
{
	fn source(&self) -> Option<&(dyn Error + 'static)>
	{
		self.source
			.as_ref()
			.map(|src_box| -> &(dyn Error + 'static) { src_box.as_ref() })
	}
}
impl From<&'static str> for EngineError
{
	fn from(string: &'static str) -> Self
	{
		Self {
			source: None,
			context: string,
		}
	}
}
impl From<Box<ValidationError>> for EngineError
{
	fn from(error: Box<ValidationError>) -> Self
	{
		panic!("{error}");
	}
}
impl From<Validated<VulkanError>> for EngineError
{
	fn from(error: Validated<VulkanError>) -> Self
	{
		Self {
			source: Some(Box::new(error.unwrap())),
			context: "a Vulkan error has occurred",
		}
	}
}
impl From<Validated<AllocateImageError>> for EngineError
{
	fn from(error: Validated<AllocateImageError>) -> Self
	{
		match error.unwrap() {
			AllocateImageError::CreateImage(source) => Self {
				context: "failed to create a Vulkan image",
				source: Some(Box::new(source)),
			},
			AllocateImageError::AllocateMemory(source) => Self {
				context: "failed to allocate memory for a Vulkan image",
				..source.into()
			},
			AllocateImageError::BindMemory(source) => Self {
				context: "failed to bind memory to a Vulkan image",
				source: Some(Box::new(source)),
			},
		}
	}
}
impl From<Validated<AllocateBufferError>> for EngineError
{
	fn from(error: Validated<AllocateBufferError>) -> Self
	{
		match error.unwrap() {
			AllocateBufferError::CreateBuffer(source) => Self {
				context: "failed to create a Vulkan buffer",
				source: Some(Box::new(source)),
			},
			AllocateBufferError::AllocateMemory(source) => Self {
				context: "failed to allocate memory for a Vulkan buffer",
				..source.into()
			},
			AllocateBufferError::BindMemory(source) => Self {
				context: "failed to bind memory to a Vulkan buffer",
				source: Some(Box::new(source)),
			},
		}
	}
}
impl From<MemoryAllocatorError> for EngineError
{
	fn from(error: MemoryAllocatorError) -> Self
	{
		let source: Box<dyn Error + Send + Sync + 'static> = match error {
			MemoryAllocatorError::AllocateDeviceMemory(inner) => Box::new(inner.unwrap()),
			other => Box::new(other),
		};

		Self {
			source: Some(source),
			context: "Vulkan memory allocation failed",
		}
	}
}
