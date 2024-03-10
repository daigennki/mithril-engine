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
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::{
	color_blend::*, depth_stencil::*, subpass::PipelineRenderingCreateInfo, viewport::Viewport, *,
};
use vulkano::pipeline::{layout::*, *};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::ShaderStages;

use super::{RenderContext, TextureSource};

const SEARCH_TEX_BYTES: &[u8] = include_bytes!("SearchTexBC4.dds");
const AREA_TEX_BYTES: &[u8] = include_bytes!("AreaTexBC5.dds");

mod compute_separate_multisample
{
	vulkano_shaders::shader! {
		ty: "compute",
		src: r"
			#version 460

			layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

			layout(binding = 0, rgba16f) uniform readonly image2DMS input_image;
			layout(binding = 1) uniform writeonly image2D output_image0;
			layout(binding = 2) uniform writeonly image2D output_image1;

			void main()
			{
				ivec2 image_coord = ivec2(gl_GlobalInvocationID.xy);
				vec4 sample0 = imageLoad(input_image, image_coord, 0);
				vec4 sample1 = imageLoad(input_image, image_coord, 1);
				imageStore(output_image0, image_coord, sample0);
				imageStore(output_image1, image_coord, sample1);
			}
		",
	}
}
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
mod fs_neighborhood_s2x
{
	vulkano_shaders::shader! {
		ty: "fragment",
		path: "src/shaders/smaa_neighborhood.frag.glsl",
		define: [("S2X", "")],
	}
}

struct ImageBundle
{
	split_input: Arc<ImageView>,
	edges_image: Arc<ImageView>, // RG
	blend_image: Arc<ImageView>, // RGBA

	input_set: Arc<PersistentDescriptorSet>, // contains separated input image
	edges_set: Arc<PersistentDescriptorSet>, // contains `edges_image`
}
impl ImageBundle
{
	/// Create a new SMAA image bundle from the given input image. The input image must only have
	/// one sample per fragment (it must have been separated if it's from a multisample image).
	fn new(
		descriptor_set_allocator: &StandardDescriptorSetAllocator,
		memory_allocator: Arc<StandardMemoryAllocator>,
		area_tex: Arc<ImageView>,
		search_tex: Arc<ImageView>,
		input_set_layout: Arc<DescriptorSetLayout>,
		edges_set_layout: Arc<DescriptorSetLayout>,
		input_image: Option<Arc<ImageView>>,
		image_extent: [u32; 3],
	) -> crate::Result<Self>
	{
		log::debug!("creating SMAA images");

		let split_input = if let Some(original) = input_image {
			original
		} else {
			let split_info = ImageCreateInfo {
				usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
				format: Format::R16G16B16A16_SFLOAT,
				extent: [image_extent[0], image_extent[1], 1],
				..Default::default()
			};
			let split = Image::new(memory_allocator.clone(), split_info.clone(), AllocationCreateInfo::default())?;
			ImageView::new_default(split)?
		};

		let edges_info = ImageCreateInfo {
			usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
			format: Format::R16G16_SFLOAT,
			extent: [image_extent[0], image_extent[1], 1],
			..Default::default()
		};
		let edges = Image::new(memory_allocator.clone(), edges_info.clone(), AllocationCreateInfo::default())?;
		let edges_image = ImageView::new_default(edges)?;

		let blend_info = ImageCreateInfo {
			usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
			format: Format::R16G16B16A16_SFLOAT,
			extent: [image_extent[0], image_extent[1], 1],
			..Default::default()
		};
		let blend = Image::new(memory_allocator, blend_info.clone(), AllocationCreateInfo::default())?;
		let blend_image = ImageView::new_default(blend)?;

		let input_writes = [
			//rt_metrics_uniform.clone(),
			WriteDescriptorSet::image_view(1, split_input.clone()),
		];
		let input_set = PersistentDescriptorSet::new(descriptor_set_allocator, input_set_layout, input_writes, [])?;

		let edges_writes = [
			//rt_metrics_uniform.clone(),
			WriteDescriptorSet::image_view(1, area_tex.clone()),
			WriteDescriptorSet::image_view(2, search_tex.clone()),
			WriteDescriptorSet::image_view(3, edges_image.clone()),
		];
		let edges_set = PersistentDescriptorSet::new(descriptor_set_allocator, edges_set_layout, edges_writes, [])?;

		Ok(Self {
			split_input,
			edges_image,
			blend_image,
			input_set,
			edges_set,
		})
	}
}
pub struct SmaaRenderer
{
	search_tex: Arc<ImageView>,
	area_tex: Arc<ImageView>,
	edges_pipeline: Arc<GraphicsPipeline>,
	blend_pipeline: Arc<GraphicsPipeline>,
	neighborhood_pipeline: Arc<GraphicsPipeline>,
	neighborhood_pipeline_s2x: Arc<GraphicsPipeline>,

	descriptor_set_allocator: StandardDescriptorSetAllocator,
	rt_metrics: Vec4,

	input_image_original: Option<Arc<ImageView>>,
	stencil_format: Format,
	stencil_image: Option<Arc<ImageView>>,
	images0: Option<ImageBundle>, // always used
	images1: Option<ImageBundle>, // only used with SMAA S2x
	blend_set: Option<Arc<PersistentDescriptorSet>>,

	multisample_split_pipeline: Arc<ComputePipeline>,
	multisample_split_set: Option<Arc<PersistentDescriptorSet>>,
}
impl SmaaRenderer
{
	pub fn new(render_ctx: &mut RenderContext) -> crate::Result<Self>
	{
		let device = render_ctx.memory_allocator.device().clone();

		let search_tex = {
			let (format, extent, _, img_raw) = super::load_texture(TextureSource::EmbeddedDds(SEARCH_TEX_BYTES)).unwrap();
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
			let (format, extent, _, img_raw) = super::load_texture(TextureSource::EmbeddedDds(AREA_TEX_BYTES)).unwrap();
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

		const STENCIL_FORMAT_CANDIDATES: [Format; 4] = [
			Format::S8_UINT,
			Format::D16_UNORM_S8_UINT,
			Format::D24_UNORM_S8_UINT,
			Format::D32_SFLOAT_S8_UINT,
		];
		let stencil_format = STENCIL_FORMAT_CANDIDATES
			.into_iter()
			.find(|format| {
				device
					.physical_device()
					.format_properties(*format)
					.unwrap()
					.optimal_tiling_features
					.contains(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
			})
			.unwrap();

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
		let rg_rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16_SFLOAT)],
			stencil_attachment_format: Some(stencil_format),
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
			subpass: Some(rg_rendering_info.into()),
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
		let rgba_rendering_info = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
			stencil_attachment_format: Some(stencil_format),
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
			depth_stencil_state: Some(depth_stencil_state.clone()),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(rgba_rendering_info.clone().into()),
			..GraphicsPipelineCreateInfo::layout(blend_pipeline_layout)
		};
		let blend_pipeline = GraphicsPipeline::new(device.clone(), None, blend_pipeline_info)?;

		let blend_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: BTreeMap::from([
				//(0, inline_uniform_binding.clone()),
				(1, tex_binding.clone()),
				(2, tex_binding.clone()),
				(3, tex_binding.clone()),
				(4, tex_binding.clone()),
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
		let no_stencil_rendering_info = PipelineRenderingCreateInfo {
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
			depth_stencil_state: None,
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(no_stencil_rendering_info.into()),
			..GraphicsPipelineCreateInfo::layout(neighborhood_pipeline_layout)
		};
		let neighborhood_pipeline = GraphicsPipeline::new(device.clone(), None, neighborhood_pipeline_info.clone())?;

		let neighborhood_pipeline_s2x_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				PipelineShaderStageCreateInfo::new(vs_neighborhood::load(device.clone())?.entry_point("main").unwrap()),
				PipelineShaderStageCreateInfo::new(fs_neighborhood_s2x::load(device.clone())?.entry_point("main").unwrap()),
			],
			..neighborhood_pipeline_info
		};
		let neighborhood_pipeline_s2x = GraphicsPipeline::new(device.clone(), None, neighborhood_pipeline_s2x_info.clone())?;

		let set_alloc_info = StandardDescriptorSetAllocatorCreateInfo {
			set_count: 4,
			..Default::default()
		};
		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), set_alloc_info);

		let storage_image_binding = DescriptorSetLayoutBinding {
			stages: ShaderStages::COMPUTE,
			..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
		};
		let color_set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, storage_image_binding.clone()),
				(1, storage_image_binding.clone()),
				(2, storage_image_binding),
			]
			.into(),
			..Default::default()
		};
		let ms_split_set_layout = DescriptorSetLayout::new(device.clone(), color_set_layout_info)?;
		let ms_split_pl_layout_info = PipelineLayoutCreateInfo {
			set_layouts: vec![ms_split_set_layout],
			..Default::default()
		};
		let ms_split_pl_layout = PipelineLayout::new(device.clone(), ms_split_pl_layout_info)?;
		let ms_split_entry_point = compute_separate_multisample::load(device.clone())?
			.entry_point("main")
			.unwrap();
		let ms_split_stage = PipelineShaderStageCreateInfo::new(ms_split_entry_point);
		let ms_split_pipeline_create_info = ComputePipelineCreateInfo::stage_layout(ms_split_stage, ms_split_pl_layout);
		let multisample_split_pipeline = ComputePipeline::new(device, None, ms_split_pipeline_create_info)?;

		Ok(Self {
			search_tex,
			area_tex,
			edges_pipeline,
			blend_pipeline,
			neighborhood_pipeline,
			neighborhood_pipeline_s2x,
			descriptor_set_allocator,
			rt_metrics: Vec4::ZERO,
			input_image_original: None,
			stencil_format,
			stencil_image: None,
			images0: None,
			images1: None,
			blend_set: None,
			multisample_split_pipeline,
			multisample_split_set: None,
		})
	}

	pub fn run(
		&mut self,
		cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
		memory_allocator: Arc<StandardMemoryAllocator>,
		input_image_original: Arc<ImageView>,
		output_image: Arc<ImageView>,
	) -> crate::Result<()>
	{
		let image_extent = output_image.image().extent();
		let extent_f32 = [image_extent[0] as f32, image_extent[1] as f32];

		// Resize images and recreate descriptor set if `input_image_original` was resized.
		let multisample_split_layout = self.multisample_split_pipeline.layout().clone();
		let edges_layout = self.edges_pipeline.layout().clone();
		let blend_layout = self.blend_pipeline.layout().clone();
		if self.input_image_original != Some(input_image_original.clone()) {
			let rt_metrics = Vec4::new(1.0 / (extent_f32[0]), 1.0 / (extent_f32[1]), extent_f32[0], extent_f32[1]);
			//let rt_metrics_uniform = WriteDescriptorSet::inline_uniform_block(0, 0, bytemuck::bytes_of(&rt_metrics).into());
			self.rt_metrics = rt_metrics;

			// TODO: separate samples into separate images if `input_image_original` has 2 samples per fragment

			let input_set_layout = edges_layout.set_layouts()[0].clone();
			let edges_set_layout = blend_layout.set_layouts()[0].clone();
			let blend_set_layout = self.neighborhood_pipeline.layout().set_layouts()[0].clone();
			let filtered_input_image =
				(input_image_original.image().samples() == SampleCount::Sample1).then(|| input_image_original.clone());
			self.images0 = Some(ImageBundle::new(
				&self.descriptor_set_allocator,
				memory_allocator.clone(),
				self.area_tex.clone(),
				self.search_tex.clone(),
				input_set_layout.clone(),
				edges_set_layout.clone(),
				filtered_input_image.clone(),
				image_extent,
			)?);

			if input_image_original.image().samples() == SampleCount::Sample2 {
				self.images1 = Some(ImageBundle::new(
					&self.descriptor_set_allocator,
					memory_allocator.clone(),
					self.area_tex.clone(),
					self.search_tex.clone(),
					input_set_layout,
					edges_set_layout,
					filtered_input_image,
					image_extent,
				)?);
			}

			let input_image0 = self.images0.as_ref().unwrap().split_input.clone();
			let input_image1 = self
				.images1
				.as_ref()
				.map_or_else(|| input_image0.clone(), |bundle| bundle.split_input.clone());
			let blend_image0 = self.images0.as_ref().unwrap().blend_image.clone();
			let blend_image1 = self
				.images1
				.as_ref()
				.map_or_else(|| blend_image0.clone(), |bundle| bundle.blend_image.clone());
			let blend_writes = [
				//rt_metrics_uniform,
				WriteDescriptorSet::image_view(1, input_image0.clone()),
				WriteDescriptorSet::image_view(2, input_image1.clone()),
				WriteDescriptorSet::image_view(3, blend_image0),
				WriteDescriptorSet::image_view(4, blend_image1),
			];
			self.blend_set = Some(PersistentDescriptorSet::new(
				&self.descriptor_set_allocator,
				blend_set_layout,
				blend_writes,
				[],
			)?);

			self.multisample_split_set = if self.images1.is_some() {
				let ms_split_set_layout = multisample_split_layout.set_layouts()[0].clone();
				let split_writes = [
					WriteDescriptorSet::image_view(0, input_image_original.clone()),
					WriteDescriptorSet::image_view(1, input_image0.clone()),
					WriteDescriptorSet::image_view(2, input_image1.clone()),
				];
				Some(PersistentDescriptorSet::new(
					&self.descriptor_set_allocator,
					ms_split_set_layout,
					split_writes,
					[],
				)?)
			} else {
				None
			};

			let stencil_info = ImageCreateInfo {
				usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
				format: self.stencil_format,
				extent: [image_extent[0], image_extent[1], 1],
				..Default::default()
			};
			let stencil = Image::new(memory_allocator, stencil_info.clone(), AllocationCreateInfo::default())?;
			self.stencil_image = Some(ImageView::new_default(stencil)?);

			self.input_image_original = Some(input_image_original);
		}

		let edges_rendering = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(self.images0.as_ref().unwrap().edges_image.clone())
			})],
			stencil_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store,
				clear_value: Some(ClearValue::Stencil(0)),
				..RenderingAttachmentInfo::image_view(self.stencil_image.clone().unwrap())
			}),
			..Default::default()
		};
		let blend_rendering = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Clear,
				store_op: AttachmentStoreOp::Store,
				clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
				..RenderingAttachmentInfo::image_view(self.images0.as_ref().unwrap().blend_image.clone())
			})],
			stencil_attachment: Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(self.stencil_image.clone().unwrap())
			}),
			..Default::default()
		};
		let neighborhood_rendering = RenderingInfo {
			color_attachments: vec![Some(RenderingAttachmentInfo {
				load_op: AttachmentLoadOp::Load,
				store_op: AttachmentStoreOp::Store,
				..RenderingAttachmentInfo::image_view(output_image)
			})],
			..Default::default()
		};
		let viewport = Viewport {
			offset: [0.0, 0.0],
			extent: extent_f32,
			depth_range: 0.0..=1.0,
		};

		// split each sample into separate images
		if let Some(split_set) = self.multisample_split_set.clone() {
			let workgroups_x = image_extent[0].div_ceil(64);
			let layout = self.multisample_split_pipeline.layout().clone();
			cb.bind_pipeline_compute(self.multisample_split_pipeline.clone())?
				.bind_descriptor_sets(PipelineBindPoint::Compute, layout, 0, split_set)?
				.dispatch([workgroups_x, image_extent[1], 1])?;
		}

		const BIND_POINT: PipelineBindPoint = PipelineBindPoint::Graphics;
		cb.begin_rendering(edges_rendering)?
			.set_viewport(0, [viewport].as_slice().into())?
			.bind_pipeline_graphics(self.edges_pipeline.clone())?
			.bind_descriptor_sets(
				BIND_POINT,
				edges_layout.clone(),
				0,
				self.images0.as_ref().unwrap().input_set.clone(),
			)?
			.push_constants(edges_layout.clone(), 0, self.rt_metrics)?
			.draw(3, 1, 0, 0)?
			.end_rendering()?;

		if let Some(images1) = self.images1.as_ref() {
			let edges_rendering1 = RenderingInfo {
				color_attachments: vec![Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(images1.edges_image.clone())
				})],
				stencil_attachment: Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Stencil(0)),
					..RenderingAttachmentInfo::image_view(self.stencil_image.clone().unwrap())
				}),
				..Default::default()
			};
			cb.begin_rendering(edges_rendering1)?
				.bind_descriptor_sets(BIND_POINT, edges_layout, 0, images1.input_set.clone())?
				.draw(3, 1, 0, 0)?
				.end_rendering()?;
		}

		cb.begin_rendering(blend_rendering)?
			.bind_pipeline_graphics(self.blend_pipeline.clone())?
			.bind_descriptor_sets(
				BIND_POINT,
				blend_layout.clone(),
				0,
				self.images0.as_ref().unwrap().edges_set.clone(),
			)?
			.push_constants(blend_layout.clone(), 0, self.rt_metrics)?
			.draw(3, 1, 0, 0)?
			.end_rendering()?;
		if let Some(images1) = self.images1.as_ref() {
			let blend_rendering1 = RenderingInfo {
				color_attachments: vec![Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Clear,
					store_op: AttachmentStoreOp::Store,
					clear_value: Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
					..RenderingAttachmentInfo::image_view(images1.blend_image.clone())
				})],
				stencil_attachment: Some(RenderingAttachmentInfo {
					load_op: AttachmentLoadOp::Load,
					store_op: AttachmentStoreOp::Store,
					..RenderingAttachmentInfo::image_view(self.stencil_image.clone().unwrap())
				}),
				..Default::default()
			};

			cb.begin_rendering(blend_rendering1)?
				.bind_descriptor_sets(BIND_POINT, blend_layout, 0, images1.edges_set.clone())?
				.draw(3, 1, 0, 0)?
				.end_rendering()?;
		}

		let neighborhood_pipeline = if self.images1.is_some() {
			self.neighborhood_pipeline_s2x.clone()
		} else {
			self.neighborhood_pipeline.clone()
		};
		let neighborhood_layout = neighborhood_pipeline.layout().clone();
		cb.begin_rendering(neighborhood_rendering)?
			.bind_pipeline_graphics(neighborhood_pipeline)?
			.bind_descriptor_sets(BIND_POINT, neighborhood_layout.clone(), 0, self.blend_set.clone().unwrap())?
			.push_constants(neighborhood_layout, 0, self.rt_metrics)?
			.draw(3, 1, 0, 0)?
			.end_rendering()?;

		Ok(())
	}
}
