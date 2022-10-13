/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::buffer::TypedBufferAccess;
use vulkano::command_buffer::{
	AutoCommandBufferBuilder, BuildError, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderPassInfo,
	CommandBufferInheritanceRenderPassType, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo, CopyError,
	ExecuteCommandsError, PipelineExecutionError, PrimaryAutoCommandBuffer, RenderPassBeginInfo, RenderPassError,
	SecondaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::DescriptorSetsCollection;
use vulkano::device::Queue;
use vulkano::pipeline::graphics::input_assembly::Index;
use vulkano::pipeline::graphics::vertex_input::VertexBuffersCollection;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::render_pass::Framebuffer;

use crate::render::pipeline;
use crate::GenericEngineError;

pub struct CommandBuffer<L>
{
	cb: AutoCommandBufferBuilder<L>,
}
impl CommandBuffer<PrimaryAutoCommandBuffer>
{
	pub fn new(queue: Arc<Queue>) -> Result<Self, GenericEngineError>
	{
		Ok(CommandBuffer {
			cb: AutoCommandBufferBuilder::primary(
				queue.device().clone(),
				queue.queue_family_index(),
				CommandBufferUsage::OneTimeSubmit,
			)?,
		})
	}

	pub fn build(self) -> Result<PrimaryAutoCommandBuffer, BuildError>
	{
		self.cb.build()
	}

	pub fn execute_secondary(&mut self, secondary: SecondaryAutoCommandBuffer) -> Result<(), ExecuteCommandsError>
	{
		self.cb.execute_commands(secondary)?;
		Ok(())
	}

	pub fn execute_secondaries(&mut self, secondaries: Vec<SecondaryAutoCommandBuffer>) -> Result<(), ExecuteCommandsError>
	{
		self.cb.execute_commands_from_vec(secondaries)?;
		Ok(())
	}

	pub fn begin_render_pass(
		&mut self, rp_begin_info: RenderPassBeginInfo, contents: SubpassContents,
	) -> Result<(), RenderPassError>
	{
		self.cb.begin_render_pass(rp_begin_info, contents)?;
		Ok(())
	}

	pub fn end_render_pass(&mut self) -> Result<(), RenderPassError>
	{
		self.cb.end_render_pass()?;
		Ok(())
	}

	pub fn next_subpass(&mut self, contents: SubpassContents) -> Result<(), RenderPassError>
	{
		self.cb.next_subpass(contents)?;
		Ok(())
	}
}
impl CommandBuffer<SecondaryAutoCommandBuffer>
{
	pub fn new(queue: Arc<Queue>, framebuffer: Option<Arc<Framebuffer>>) -> Result<Self, GenericEngineError>
	{
		let inheritance = CommandBufferInheritanceInfo {
			render_pass: framebuffer.and_then(|fb| {
				Some(CommandBufferInheritanceRenderPassType::BeginRenderPass(CommandBufferInheritanceRenderPassInfo {
					subpass: fb.render_pass().clone().first_subpass(),
					framebuffer: Some(fb),
				}))
			}),
			..Default::default()
		};
		let new_cb = AutoCommandBufferBuilder::secondary(
			queue.device().clone(),
			queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
			inheritance,
		)?;

		Ok(CommandBuffer { cb: new_cb })
	}

	pub fn build(self) -> Result<SecondaryAutoCommandBuffer, vulkano::command_buffer::BuildError>
	{
		self.cb.build()
	}
}
impl<L> CommandBuffer<L>
{
	pub fn bind_pipeline(&mut self, pipeline_to_bind: &pipeline::Pipeline)
	{
		pipeline_to_bind.bind(&mut self.cb);
	}

	/// Bind the given descriptor sets to the currently bound pipeline.
	/// This will fail if there is no pipeline currently bound.
	pub fn bind_descriptor_set<S>(&mut self, first_set: u32, descriptor_sets: S) -> Result<(), PipelineExecutionError>
	where
		S: DescriptorSetsCollection,
	{
		let pipeline_layout = self
			.cb
			.state()
			.pipeline_graphics()
			.ok_or(PipelineExecutionError::PipelineNotBound)?
			.layout()
			.clone();
		self.cb
			.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout, first_set, descriptor_sets);
		Ok(())
	}

	pub fn bind_vertex_buffers<V>(&mut self, first_binding: u32, vertex_buffers: V)
	where
		V: VertexBuffersCollection,
	{
		self.cb.bind_vertex_buffers(first_binding, vertex_buffers);
	}

	pub fn bind_index_buffer<Ib, I>(&mut self, index_buffer: Arc<Ib>)
	where
		Ib: TypedBufferAccess<Content = [I]> + 'static,
		I: Index + 'static,
	{
		self.cb.bind_index_buffer(index_buffer);
	}

	pub fn draw(
		&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32,
	) -> Result<(), PipelineExecutionError>
	{
		self.cb
			.draw(vertex_count, instance_count, first_vertex, first_instance)?;
		Ok(())
	}

	pub fn draw_indexed(
		&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32,
	) -> Result<(), PipelineExecutionError>
	{
		self.cb
			.draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance)?;
		Ok(())
	}

	pub fn set_viewport<I>(&mut self, first_viewport: u32, viewports: I)
	where
		I: IntoIterator<Item = Viewport>,
	{
		self.cb.set_viewport(first_viewport, viewports);
	}

	pub fn copy_buffer_to_image(&mut self, copy_buffer_to_image_info: CopyBufferToImageInfo) -> Result<(), CopyError>
	{
		self.cb.copy_buffer_to_image(copy_buffer_to_image_info)?;
		Ok(())
	}

	pub fn copy_buffer(&mut self, copy_buffer_info: CopyBufferInfo) -> Result<(), CopyError>
	{
		self.cb.copy_buffer(copy_buffer_info)?;
		Ok(())
	}
}
