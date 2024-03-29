/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use glam::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::command_buffer::*;
use vulkano::descriptor_set::{allocator::*, layout::*, *};
use vulkano::device::DeviceOwned;
use vulkano::format::*;
use vulkano::image::{sampler::*, view::ImageView, *};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::graphics::{
	color_blend::*, depth_stencil::*, subpass::PipelineRenderingCreateInfo, viewport::Viewport, *,
};
use vulkano::pipeline::{layout::*, *};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

use super::{RenderContext, TextureSource};

const SEARCH_TEX_BYTES: &[u8] = include_bytes!("SearchTexBC4.dds");
const AREA_TEX_BYTES: &[u8] = include_bytes!("AreaTexBC5.dds");

mod vs_edges
{
	vulkano_shaders::shader! {
		ty: "vertex",
		path: "src/shaders/smaa_edge.vert.glsl",
	}
}
mod fs_edges
{
	vulkano_shaders::shader! {
		ty: "fragment",
		path: "src/shaders/smaa_edge.frag.glsl",
	}
}
mod vs_blend
{
	vulkano_shaders::shader! {
		ty: "vertex",
		path: "src/shaders/smaa_blend.vert.glsl",
	}
}
mod fs_blend
{
	vulkano_shaders::shader! {
		ty: "fragment",
		path: "src/shaders/smaa_blend.frag.glsl",
	}
}
mod vs_neighborhood
{
	vulkano_shaders::shader! {
		ty: "vertex",
		path: "src/shaders/smaa_neighborhood.vert.glsl",
	}
}
mod fs_neighborhood
{
	vulkano_shaders::shader! {
		ty: "fragment",
		path: "src/shaders/smaa_neighborhood.frag.glsl",
	}
}

pub struct SmaaRenderer
{
	search_tex: Arc<ImageView>,
	area_tex: Arc<ImageView>,
	edges_pipeline: Arc<GraphicsPipeline>,
	blend_pipeline: Arc<GraphicsPipeline>,
	neighborhood_pipeline: Arc<GraphicsPipeline>,

	descriptor_set_allocator: StandardDescriptorSetAllocator,
	rt_metrics: Vec4,

	edges_image: Option<Arc<ImageView>>,
	blend_image: Option<Arc<ImageView>>,
	input_set: Option<Arc<PersistentDescriptorSet>>,
	edges_set: Option<Arc<PersistentDescriptorSet>>,
	blend_set: Option<Arc<PersistentDescriptorSet>>,
}
impl SmaaRenderer
{
	pub fn new(render_ctx: &mut RenderContext) -> crate::Result<Self>
	{
		let device = render_ctx.memory_allocator.device().clone();

		let search_tex = {
			let (format, extent, _, img_raw) = TextureSource::Bytes(SEARCH_TEX_BYTES).load().unwrap();
			let image_info = ImageCreateInfo {
				format,
				extent: [extent[0], extent[1], 1],
				usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
				..Default::default()
			};
			let image = render_ctx.new_image(&img_raw, image_info)?;
			ImageView::new_default(image)?
		};
		let area_tex = {
			let (format, extent, _, img_raw) = TextureSource::Bytes(AREA_TEX_BYTES).load().unwrap();
			let image_info = ImageCreateInfo {
				format,
				extent: [extent[0], extent[1], 1],
				usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
				..Default::default()
			};
			let image = render_ctx.new_image(&img_raw, image_info)?;
			ImageView::new_default(image)?
		};

		let color_blend_state = ColorBlendState::with_attachment_states(1, Default::default());

		let write_stencil_op_state = StencilOpState {
			ops: StencilOps {
				pass_op: StencilOp::IncrementAndClamp,
				compare_op: CompareOp::Always,
				..Default::default()
			},
			..Default::default()
		};
		let depth_stencil_state_write_stencil = DepthStencilState {
			stencil: Some(StencilState {
				front: write_stencil_op_state,
				back: write_stencil_op_state,
			}),
			..Default::default()
		};
		let stencil_op_state = StencilOpState {
			ops: StencilOps {
				compare_op: CompareOp::Less,
				..Default::default()
			},
			reference: 0,
			..Default::default()
		};
		let depth_stencil_state = DepthStencilState {
			stencil: Some(StencilState {
				front: stencil_op_state,
				back: stencil_op_state,
			}),
			..Default::default()
		};

		/*let inline_uniform_binding = DescriptorSetLayoutBinding {
			descriptor_count: 16,
			stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::InlineUniformBlock)
		};*/
		let push_constant_range = PushConstantRange {
			stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
			offset: 0,
			size: 16,
		};
		let sampler_info = SamplerCreateInfo {
			mag_filter: Filter::Linear,
			min_filter: Filter::Linear,
			..Default::default()
		};
		let sampler = Sampler::new(device.clone(), sampler_info)?;
		let tex_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::FRAGMENT,
			immutable_samplers: vec![sampler],
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
		};
		let input_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: BTreeMap::from([
				//(0, inline_uniform_binding.clone()),
				(1, tex_binding.clone()),
			]),
			..Default::default()
		};
		let input_set_layout = DescriptorSetLayout::new(device.clone(), input_set_layout_info)?;
		let edges_pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![input_set_layout],
			push_constant_ranges: vec![push_constant_range],
			..Default::default()
		};
		let edges_pipeline_layout = PipelineLayout::new(device.clone(), edges_pipeline_layout_info)?;
		let edges_rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R8G8_UNORM)],
			stencil_attachment_format: Some(render_ctx.depth_stencil_format),
			..Default::default()
		};
		let edges_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				PipelineShaderStageCreateInfo::new(vs_edges::load(device.clone())?.entry_point("main").unwrap()),
				PipelineShaderStageCreateInfo::new(fs_edges::load(device.clone())?.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(Default::default()),
			input_assembly_state: Some(Default::default()),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(Default::default()),
			multisample_state: Some(Default::default()),
			color_blend_state: Some(color_blend_state.clone()),
			depth_stencil_state: Some(depth_stencil_state_write_stencil),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(edges_rendering_info.clone().into()),
			..GraphicsPipelineCreateInfo::layout(edges_pipeline_layout)
		};
		let edges_pipeline = GraphicsPipeline::new(device.clone(), None, edges_pipeline_info)?;

		let edges_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: BTreeMap::from([
				//(0, inline_uniform_binding.clone()),
				(1, tex_binding.clone()),
				(2, tex_binding.clone()),
				(3, tex_binding.clone()),
			]),
			..Default::default()
		};
		let edges_set_layout = DescriptorSetLayout::new(device.clone(), edges_set_layout_info)?;
		let blend_pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![edges_set_layout],
			push_constant_ranges: vec![push_constant_range],
			..Default::default()
		};
		let blend_pipeline_layout = PipelineLayout::new(device.clone(), blend_pipeline_layout_info)?;
		let blend_rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R8G8B8A8_UNORM)],
			stencil_attachment_format: Some(render_ctx.depth_stencil_format),
			..Default::default()
		};
		let blend_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				PipelineShaderStageCreateInfo::new(vs_blend::load(device.clone())?.entry_point("main").unwrap()),
				PipelineShaderStageCreateInfo::new(fs_blend::load(device.clone())?.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(Default::default()),
			input_assembly_state: Some(Default::default()),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(Default::default()),
			multisample_state: Some(Default::default()),
			color_blend_state: Some(color_blend_state.clone()),
			depth_stencil_state: Some(depth_stencil_state),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(blend_rendering_info.into()),
			..GraphicsPipelineCreateInfo::layout(blend_pipeline_layout)
		};
		let blend_pipeline = GraphicsPipeline::new(device.clone(), None, blend_pipeline_info)?;

		let blend_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: BTreeMap::from([
				//(0, inline_uniform_binding.clone()),
				(1, tex_binding.clone()),
				(2, tex_binding.clone()),
			]),
			..Default::default()
		};
		let blend_set_layout = DescriptorSetLayout::new(device.clone(), blend_set_layout_info)?;
		let neighborhood_pipeline_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![blend_set_layout],
			push_constant_ranges: vec![push_constant_range],
			..Default::default()
		};
		let neighborhood_pipeline_layout = PipelineLayout::new(device.clone(), neighborhood_pipeline_layout_info)?;
		let neighborhood_rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
			..Default::default()
		};
		let neighborhood_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				PipelineShaderStageCreateInfo::new(vs_neighborhood::load(device.clone())?.entry_point("main").unwrap()),
				PipelineShaderStageCreateInfo::new(fs_neighborhood::load(device.clone())?.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(Default::default()),
			input_assembly_state: Some(Default::default()),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(Default::default()),
			multisample_state: Some(Default::default()),
			color_blend_state: Some(color_blend_state),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(neighborhood_rendering_info.into()),
			..GraphicsPipelineCreateInfo::layout(neighborhood_pipeline_layout)
		};
		let neighborhood_pipeline = GraphicsPipeline::new(device.clone(), None, neighborhood_pipeline_info.clone())?;

		let set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 4,
			..Default::default()
		};
		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), set_alloc_info);

		Ok(Self {
			search_tex,
			area_tex,
			edges_pipeline,
			blend_pipeline,
			neighborhood_pipeline,
			descriptor_set_allocator,
			rt_metrics: Vec4::ZERO,
			edges_image: None,
			blend_image: None,
			input_set: None,
			edges_set: None,
			blend_set: None,
		})
	}

	pub fn run(
		&mut self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		memory_allocator: Arc<StandardMemoryAllocator>,
		input_image: Arc<ImageView>,
		stencil_image: Arc<ImageView>,
		output_image: Arc<ImageView>,
	) -> crate::Result<()>
	{
		assert!(input_image.image().samples() == SampleCount::Sample1);

		let image_extent = output_image.image().extent();
		let extent_f32 = [image_extent[0] as f32, image_extent[1] as f32];

		// Resize images and recreate descriptor set if the output image was resized.
		let edges_layout = self.edges_pipeline.layout().clone();
		let blend_layout = self.blend_pipeline.layout().clone();
		let neighborhood_layout = self.neighborhood_pipeline.layout().clone();
		if Some(image_extent) != self.edges_image.as_ref().map(|view| view.image().extent()) {
			self.rt_metrics = Vec4::new(1.0 / (extent_f32[0]), 1.0 / (extent_f32[1]), extent_f32[0], extent_f32[1]);
			//let rt_metrics_uniform = WriteDescriptorSet::inline_uniform_block(0, 0, bytemuck::bytes_of(&rt_metrics).into());

			let input_set_layout = edges_layout.set_layouts()[0].clone();
			let input_writes = [
				//rt_metrics_uniform.clone(),
				WriteDescriptorSet::image_view(1, input_image.clone()),
			];
			let input_set = PersistentDescriptorSet::new(&self.descriptor_set_allocator, input_set_layout, input_writes, [])?;
			self.input_set = Some(input_set);

			let edges_image_info = ImageCreateInfo {
				usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
				format: Format::R8G8_UNORM,
				extent: image_extent,
				..Default::default()
			};
			let edges = Image::new(memory_allocator.clone(), edges_image_info, AllocationCreateInfo::default())?;
			let edges_image = ImageView::new_default(edges)?;
			self.edges_image = Some(edges_image.clone());

			// We use the output image as the edges image since the output image won't be needed
			// until the neighborhood pass.
			let edges_set_layout = blend_layout.set_layouts()[0].clone();
			let edges_writes = [
				//rt_metrics_uniform.clone(),
				WriteDescriptorSet::image_view(1, self.area_tex.clone()),
				WriteDescriptorSet::image_view(2, self.search_tex.clone()),
				WriteDescriptorSet::image_view(3, edges_image),
			];
			let edges_set = PersistentDescriptorSet::new(&self.descriptor_set_allocator, edges_set_layout, edges_writes, [])?;
			self.edges_set = Some(edges_set);

			let blend_image_info = ImageCreateInfo {
				usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
				format: Format::R8G8B8A8_UNORM,
				extent: image_extent,
				..Default::default()
			};
			let blend = Image::new(memory_allocator, blend_image_info, AllocationCreateInfo::default())?;
			let blend_image = ImageView::new_default(blend)?;
			self.blend_image = Some(blend_image.clone());

			let blend_set_layout = neighborhood_layout.set_layouts()[0].clone();
			let blend_writes = [
				//rt_metrics_uniform,
				WriteDescriptorSet::image_view(1, input_image),
				WriteDescriptorSet::image_view(2, blend_image),
			];
			self.blend_set = Some(PersistentDescriptorSet::new(
				&self.descriptor_set_allocator,
				blend_set_layout,
				blend_writes,
				[],
			)?);
		}

		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: extent_f32,
			depth_range: 0.0..=1.0,
		};
		let edges_rendering = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store,
				clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
				..RenderingAttachmentInfo::image_view(self.edges_image.clone().unwrap())
			})],
			stencil_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store,
				clear_value: Some(ClearValue::Stencil(0)),
				..RenderingAttachmentInfo::image_view(stencil_image.clone())
			}),
			..Default::default()
		};
		let blend_rendering = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store,
				clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
				..RenderingAttachmentInfo::image_view(self.blend_image.clone().unwrap())
			})],
			stencil_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				..RenderingAttachmentInfo::image_view(stencil_image)
			}),
			..Default::default()
		};
		let neighborhood_rendering = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(output_image)
			})],
			..Default::default()
		};

		const BIND_POINT: PipelineBindPoint = PipelineBindPoint::Graphics;
		cb.set_viewport(0, smallvec::smallvec![viewport])?
			.bind_pipeline_graphics(self.edges_pipeline.clone())?
			.push_constants(edges_layout.clone(), 0, self.rt_metrics)?
			.bind_descriptor_sets(BIND_POINT, edges_layout, 0, self.input_set.clone().unwrap())?
			.begin_rendering(edges_rendering)?
			.draw(3, 1, 0, 0)?
			.end_rendering()?
			.bind_pipeline_graphics(self.blend_pipeline.clone())?
			.push_constants(blend_layout.clone(), 0, self.rt_metrics)?
			.bind_descriptor_sets(BIND_POINT, blend_layout, 0, self.edges_set.clone().unwrap())?
			.begin_rendering(blend_rendering)?
			.draw(3, 1, 0, 0)?
			.end_rendering()?
			.bind_pipeline_graphics(self.neighborhood_pipeline.clone())?
			.push_constants(neighborhood_layout.clone(), 0, self.rt_metrics)?
			.bind_descriptor_sets(BIND_POINT, neighborhood_layout, 0, self.blend_set.clone().unwrap())?
			.begin_rendering(neighborhood_rendering)?
			.draw(3, 1, 0, 0)?
			.end_rendering()?;

		Ok(())
	}
}
