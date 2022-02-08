/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::sync::Arc;
use vulkano::buffer::ImmutableBuffer;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::persistent::PersistentDescriptorSet;
use vulkano::descriptor_set::layout::DescriptorDesc;
use vulkano::descriptor_set::layout::DescriptorType;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::shader::ShaderStages;
use super::rendercontext::texture::Texture;
use super::rendercontext::RenderContext;

pub struct Image
{
	tex: Texture,
	transform_buf: Arc<ImmutableBuffer<glam::Mat4>>,
	descriptor_set: Arc<PersistentDescriptorSet>
}
impl Image
{
	pub fn new(render_ctx: &mut RenderContext, path: &std::path::Path) 
		-> Result<Image, Box<dyn std::error::Error>>
	{
		let transformation = glam::Mat4::IDENTITY;
		let buf_usage = BufferUsage {
			uniform_buffer: true,
			..BufferUsage::none()
		};
		let transform_buf = render_ctx.new_buffer(transformation, buf_usage)?;

		// texture
		let tex = render_ctx.new_texture(path)?;

		// create descriptor set layout
		// TODO: use this for other UI elements too (use a common trait?)
		let set_layout = DescriptorSetLayout::new(render_ctx.device(), [
			None, None, None, None, // skip sampler slots
			Some(DescriptorDesc{ // transformation UBO
				ty: DescriptorType::UniformBuffer, 
				descriptor_count: 1, 
				variable_count: false, 
				stages: ShaderStages{ vertex: true, ..ShaderStages::none() },
				immutable_samplers: [].into()
			}),
			None, None, None, // skip other UBO slots
			Some(DescriptorDesc{ // texture
				ty: DescriptorType::SampledImage,
				descriptor_count: 1,
				variable_count: false,
				stages: ShaderStages{ fragment: true, ..ShaderStages::none() },
				immutable_samplers: [].into()
			})
		])?;

		// create descriptor set
		let descriptor_set = PersistentDescriptorSet::new(set_layout, [
			WriteDescriptorSet::buffer(4, transform_buf.clone()),
			WriteDescriptorSet::image_view(8, tex.clone_view())
		])?;

		Ok(Image{
			tex: render_ctx.new_texture(path)?,
			transform_buf: transform_buf,
			descriptor_set: descriptor_set
		})
	}

	pub fn draw(&mut self, render_ctx: &mut RenderContext) -> Result<(), super::rendercontext::CommandBufferNotBuilding>
	{
		render_ctx.bind_ui_descriptor_set(0, self.descriptor_set.clone())
	}
}
