/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
pub mod pbr;

use glam::*;
use serde::Deserialize;
use std::any::{Any, TypeId};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::device::{Device, DeviceOwned};
use vulkano::format::Format;
use vulkano::image::{view::ImageView, *};
use vulkano::pipeline::graphics::{
	color_blend::*, depth_stencil::*, multisample::*, rasterization::*, subpass::*, vertex_input::VertexInputState,
	GraphicsPipelineCreateInfo,
};
use vulkano::pipeline::*;
use vulkano::shader::{ShaderModule, SpecializationConstant};
use vulkano::{Validated, VulkanError};

use crate::render::RenderContext;

pub mod vs_3d_common
{
	vulkano_shaders::shader! {
		ty: "vertex",
		path: "src/shaders/basic_3d.vert.glsl",
	}
}

/// A material used by meshes to set shader parameters.
#[typetag::deserialize]
pub trait Material: Any + Send + Sync
{
	/// Return the list of colors/image files that should be loaded into a texture and then written
	/// into the descriptor set image view array.
	///
	/// The first entry in the `Vec` returned *must* be the "base color" image, with an alpha
	/// channel representing transparency.
	fn get_shader_inputs(&self) -> Vec<ShaderInput>;

	fn blend_mode(&self) -> BlendMode;
}

#[derive(Debug)]
pub enum ShaderInput
{
	Color(ColorInput),
	Greyscale(GreyscaleInput),
}
impl ShaderInput
{
	pub fn into_texture(self, path_prefix: &Path, render_ctx: &mut RenderContext) -> crate::Result<Arc<ImageView>>
	{
		match self {
			Self::Color(input) => input.into_texture(path_prefix, render_ctx),
			Self::Greyscale(input) => input.into_texture(path_prefix, render_ctx),
		}
	}
}
impl From<ColorInput> for ShaderInput
{
	fn from(input: ColorInput) -> Self
	{
		Self::Color(input)
	}
}
impl From<GreyscaleInput> for ShaderInput
{
	fn from(input: GreyscaleInput) -> Self
	{
		Self::Greyscale(input)
	}
}

/// A representation of the possible shader color inputs, like those on the shader nodes in Blender.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum ColorInput
{
	/// Single color value. This is in linear color space (*not* gamma corrected) for consistency
	/// with Blender's RGB color picker.
	/// (https://docs.blender.org/manual/en/latest/interface/controls/templates/color_picker.html)
	Color(Vec4),

	/// A texture image file.
	Texture(PathBuf),
}
impl ColorInput
{
	pub fn into_texture(self, path_prefix: &Path, render_ctx: &mut RenderContext) -> crate::Result<Arc<ImageView>>
	{
		match self {
			Self::Color(color) => new_single_color_texture(render_ctx, color),
			Self::Texture(tex_path) => render_ctx.new_texture(&path_prefix.join(tex_path)),
		}
	}
}

/// A representation of the possible shader greyscale inputs, like those on the shader nodes in Blender.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum GreyscaleInput
{
	Value(f32),
	Texture(PathBuf),
}
impl GreyscaleInput
{
	pub fn into_texture(self, path_prefix: &Path, render_ctx: &mut RenderContext) -> crate::Result<Arc<ImageView>>
	{
		match self {
			Self::Value(value) => new_single_color_texture(render_ctx, Vec4::new(value, value, value, 1.0)),
			Self::Texture(tex_path) => render_ctx.new_texture(&path_prefix.join(tex_path)),
		}
	}
}

// If the input is a single color, make a 1x1 RGBA texture with just the color.
fn new_single_color_texture(render_ctx: &mut RenderContext, color: Vec4) -> crate::Result<Arc<ImageView>>
{
	let create_info = ImageCreateInfo {
		format: Format::R32G32B32A32_SFLOAT,
		extent: [1, 1, 1],
		usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
		..Default::default()
	};
	let image = render_ctx.new_image(&[color], create_info)?;
	Ok(ImageView::new_default(image)?)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum BlendMode
{
	/// Don't blend (ignore alpha).
	Opaque,
	/// Blend with Order-Independent Transparency, using the base color input's alpha channel.
	AlphaBlend,
}
impl Default for BlendMode
{
	fn default() -> Self
	{
		Self::Opaque
	}
}

pub type ShaderLoader = &'static (dyn Fn(Arc<Device>) -> Result<Arc<ShaderModule>, Validated<VulkanError>> + Send + Sync);

#[derive(Copy, Clone)]
pub struct MaterialPipelineConfig
{
	// `name` is only for debugging purposes!! Use `type_id` to uniquely identify the material type.
	pub name: &'static str,

	// `TypeId` currently can't be `const`, so we use a getter function instead.
	pub type_id: &'static (dyn Fn() -> TypeId + Send + Sync),

	pub vertex_shader: ShaderLoader,
	pub fragment_shader: ShaderLoader,
}
inventory::collect!(MaterialPipelineConfig);
impl MaterialPipelineConfig
{
	pub(crate) fn into_pipelines(
		&'static self,
		rasterization_samples: SampleCount,
		depth_stencil_format: Format,
		pipeline_layout: Arc<PipelineLayout>,
	) -> crate::Result<MaterialPipelines>
	{
		let device = pipeline_layout.device().clone();

		let vs = (self.vertex_shader)(device.clone())?;
		let vs_stage = PipelineShaderStageCreateInfo::new(vs.entry_point("main").unwrap());
		let fs = (self.fragment_shader)(device.clone())?;

		let vertex_input_state = VertexInputState {
			bindings: (0..).zip(crate::render::model::VERTEX_BINDINGS).collect(),
			attributes: (0..).zip(crate::render::model::VERTEX_ATTRIBUTES).collect(),
			..Default::default()
		};
		let rasterization_state = RasterizationState {
			cull_mode: CullMode::Back,
			..Default::default()
		};
		let multisample_state = MultisampleState {
			rasterization_samples,
			..Default::default()
		};

		let rendering_formats = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
			depth_attachment_format: Some(depth_stencil_format),
			..Default::default()
		};

		// Create the opaque pass pipeline.
		let opaque_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				vs_stage.clone(),
				PipelineShaderStageCreateInfo::new(fs.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(vertex_input_state.clone()),
			input_assembly_state: Some(Default::default()),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(rasterization_state.clone()),
			multisample_state: Some(multisample_state),
			depth_stencil_state: Some(DepthStencilState {
				depth: Some(DepthState {
					write_enable: true,
					compare_op: CompareOp::Less,
				}),
				..Default::default()
			}),
			color_blend_state: Some(ColorBlendState {
				attachments: vec![ColorBlendAttachmentState::default()],
				..Default::default()
			}),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(rendering_formats.into()),
			..GraphicsPipelineCreateInfo::layout(pipeline_layout.clone())
		};
		let opaque_pipeline = GraphicsPipeline::new(device.clone(), None, opaque_pipeline_info)?;

		// Create the transparency pass pipeline. The fragment shader must have a boolean
		// specialization constant with ID 0, which specifies if the shader is for an OIT pass.
		let oit_depth_stencil_state = DepthStencilState {
			depth: Some(DepthState {
				write_enable: false,
				compare_op: CompareOp::Less,
			}),
			stencil: Some(StencilState {
				front: StencilOpState {
					ops: StencilOps {
						pass_op: StencilOp::IncrementAndClamp,
						compare_op: CompareOp::Always,
						..Default::default()
					},
					..Default::default()
				},
				..Default::default()
			}),
			..Default::default()
		};

		// accum (all additive) and revealage (dst = dst * (1 - src)) blending
		let oit_color_blend_state = ColorBlendState {
			attachments: vec![
				ColorBlendAttachmentState {
					blend: Some(AttachmentBlend {
						src_color_blend_factor: BlendFactor::One,
						dst_color_blend_factor: BlendFactor::One,
						src_alpha_blend_factor: BlendFactor::One,
						dst_alpha_blend_factor: BlendFactor::One,
						..Default::default()
					}),
					..Default::default()
				},
				ColorBlendAttachmentState {
					blend: Some(AttachmentBlend {
						src_color_blend_factor: BlendFactor::Zero,
						dst_color_blend_factor: BlendFactor::OneMinusSrcColor,
						..Default::default()
					}),
					..Default::default()
				},
			],
			..Default::default()
		};

		let oit_rendering_formats = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT), Some(Format::R8_UNORM)],
			depth_attachment_format: Some(depth_stencil_format),
			stencil_attachment_format: Some(depth_stencil_format),
			..Default::default()
		};

		let fs_oit = fs.specialize([(0, SpecializationConstant::Bool(true))].into_iter().collect())?;
		let oit_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				vs_stage,
				PipelineShaderStageCreateInfo::new(fs_oit.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(vertex_input_state),
			input_assembly_state: Some(Default::default()),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(rasterization_state),
			multisample_state: Some(multisample_state),
			depth_stencil_state: Some(oit_depth_stencil_state),
			color_blend_state: Some(oit_color_blend_state),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(oit_rendering_formats.into()),
			..GraphicsPipelineCreateInfo::layout(pipeline_layout)
		};
		let oit_pipeline = GraphicsPipeline::new(device, None, oit_pipeline_info)?;

		Ok(MaterialPipelines {
			opaque_pipeline,
			oit_pipeline,
			config: self,
		})
	}
}

pub(crate) struct MaterialPipelines
{
	pub opaque_pipeline: Arc<GraphicsPipeline>,
	pub oit_pipeline: Arc<GraphicsPipeline>,
	pub config: &'static MaterialPipelineConfig,
}
impl MaterialPipelines
{
	pub(crate) fn recreate(&mut self, rasterization_samples: SampleCount) -> crate::Result<()>
	{
		let depth_stencil_format = match self.opaque_pipeline.subpass() {
			PipelineSubpassType::BeginRendering(rendering_info) => rendering_info.depth_attachment_format.unwrap(),
			_ => unreachable!(),
		};
		let pipeline_layout = self.opaque_pipeline.layout().clone();
		*self = self
			.config
			.into_pipelines(rasterization_samples, depth_stencil_format, pipeline_layout)?;
		Ok(())
	}
}
