 /* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::command_buffer::{ AutoCommandBufferBuilder, CommandBufferUsage, DrawError, CommandBufferInheritanceInfo };
use vulkano::command_buffer::{ SubpassContents, RenderPassError, CheckPipelineError, CommandBufferInheritanceRenderPassType };
use vulkano::command_buffer::{ PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer, CommandBufferInheritanceRenderPassInfo };
use vulkano::pipeline::{ Pipeline, PipelineBindPoint };
use vulkano::pipeline::graphics::vertex_input::VertexBuffersCollection;
use vulkano::pipeline::graphics::input_assembly::Index;
use vulkano::descriptor_set::DescriptorSetsCollection;
use vulkano::buffer::TypedBufferAccess;
use vulkano::render_pass::Framebuffer;

use crate::render::pipeline;


pub struct CommandBuffer<L>
{
	cb: AutoCommandBufferBuilder<L>
}
impl CommandBuffer<PrimaryAutoCommandBuffer>
{
	pub fn new(device: Arc<vulkano::device::Device>)
		-> Result<Self, Box<dyn std::error::Error>>
	{
		let q_fam = device.active_queue_families().next()
			.ok_or("There are no active queue families in the logical device!")?;
		let new_cb = AutoCommandBufferBuilder::primary(device.clone(), q_fam, CommandBufferUsage::OneTimeSubmit)?;
		
		Ok(CommandBuffer{ cb: new_cb })
	}

	pub fn build(self) -> Result<PrimaryAutoCommandBuffer, vulkano::command_buffer::BuildError>
	{
		self.cb.build()	
	}

	pub fn begin_render_pass(&mut self, rp_begin_info: vulkano::command_buffer::RenderPassBeginInfo) 
		-> Result<(), RenderPassError>
	{
		self.cb.begin_render_pass(rp_begin_info, SubpassContents::Inline)?;
		Ok(())
	}

	pub fn end_render_pass(&mut self) -> Result<(), RenderPassError>
	{
		self.cb.end_render_pass()?;
		Ok(())
	}
}
impl CommandBuffer<SecondaryAutoCommandBuffer>
{
	pub fn new(device: Arc<vulkano::device::Device>, framebuffer: Arc<Framebuffer>)
		-> Result<Self, Box<dyn std::error::Error>>
	{
		let q_fam = device.active_queue_families().next()
			.ok_or("There are no active queue families in the logical device!")?;
		let render_pass = framebuffer.render_pass().clone();
		let subpass = render_pass.first_subpass();
		let inheritance = CommandBufferInheritanceInfo{
			render_pass: Some(
				CommandBufferInheritanceRenderPassType::BeginRenderPass(CommandBufferInheritanceRenderPassInfo{
					subpass: subpass,
					framebuffer: Some(framebuffer)
				})
			),
			..Default::default()
		};
		let new_cb = AutoCommandBufferBuilder::secondary(
			device.clone(), q_fam, CommandBufferUsage::OneTimeSubmit, inheritance
		)?;

		Ok(CommandBuffer{ cb: new_cb })
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
	pub fn bind_descriptor_set<S>(&mut self, first_set: u32, descriptor_sets: S) -> Result<(), CheckPipelineError>
		where S: DescriptorSetsCollection
	{
		let pipeline_layout = self.cb.state()
			.pipeline_graphics()
			.ok_or(CheckPipelineError::PipelineNotBound)?
			.layout()
			.clone();
		self.cb.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout, first_set, descriptor_sets);
		Ok(())
	}
	
	pub fn bind_vertex_buffers<V>(&mut self, first_binding: u32, vertex_buffers: V)
		where V: VertexBuffersCollection
	{
		self.cb.bind_vertex_buffers(first_binding, vertex_buffers);
	}

	pub fn bind_index_buffers<Ib, I>(&mut self, index_buffer: Arc<Ib>)
		where 
			Ib: TypedBufferAccess<Content = [I]> + 'static,
			I: Index + 'static
	{
		self.cb.bind_index_buffer(index_buffer);
	}

	pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32)
		-> Result<(), DrawError>
	{
		self.cb.draw(vertex_count, instance_count, first_vertex, first_instance)?;
		Ok(())
	}
}

