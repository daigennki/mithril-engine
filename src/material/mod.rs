/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod pbr;

use glam::*;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::device::{Device, DeviceOwned};
use vulkano::format::Format;
use vulkano::image::{view::ImageView, ImageCreateInfo, ImageUsage};
use vulkano::pipeline::{
	graphics::{
		color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState},
		depth_stencil::{CompareOp, DepthState, DepthStencilState, StencilOp, StencilOpState, StencilOps, StencilState},
		rasterization::{CullMode, RasterizationState},
		subpass::PipelineRenderingCreateInfo,
		vertex_input::VertexInputState,
		GraphicsPipelineCreateInfo,
	},
	DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::ShaderModule;

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
pub trait Material: Send + Sync
{
	fn material_name(&self) -> &'static str;

	/// Return the list of colors/image files that should be loaded into a texture and then written
	/// into the descriptor set image view array.
	///
	/// The first entry in the `Vec` returned *must* be the "base color" image, with an alpha
	/// channel representing transparency.
	fn get_shader_inputs(&self) -> Vec<ShaderInput>;

	fn has_transparency(&self) -> bool;

	fn load_shaders(&self, vk_dev: Arc<Device>) -> crate::Result<MaterialPipelineConfig>;
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

pub enum MaterialTransparencyMode
{
	NoTransparency,
	Blend(AttachmentBlend),
	OIT(Arc<ShaderModule>),
}
impl MaterialTransparencyMode
{
	fn into_blend_or_shader(self) -> (Option<AttachmentBlend>, Option<Arc<ShaderModule>>)
	{
		match self {
			Self::NoTransparency => (None, None),
			Self::Blend(blend) => (Some(blend), None),
			Self::OIT(fs_oit) => (None, Some(fs_oit.clone())),
		}
	}
}

pub struct MaterialPipelines
{
	pub opaque_pipeline: Arc<GraphicsPipeline>,
	pub oit_pipeline: Option<Arc<GraphicsPipeline>>, // Optional transparency pipeline may also be specified.
}

pub struct MaterialPipelineConfig
{
	pub vertex_shader: Arc<ShaderModule>,
	pub fragment_shader: Arc<ShaderModule>,
	pub transparency: MaterialTransparencyMode,
}
impl MaterialPipelineConfig
{
	pub fn into_pipelines(
		self,
		depth_format: Format,
		pipeline_layout: Arc<PipelineLayout>,
		pipeline_layout_oit: Arc<PipelineLayout>,
	) -> crate::Result<MaterialPipelines>
	{
		let device = pipeline_layout.device().clone();

		let vs_stage = PipelineShaderStageCreateInfo::new(self.vertex_shader.entry_point("main").unwrap());

		let (attachment_blend, fs_oit) = self.transparency.into_blend_or_shader();

		let vertex_input_state = VertexInputState {
			bindings: (0..).zip(crate::render::model::VERTEX_BINDINGS).collect(),
			attributes: (0..).zip(crate::render::model::VERTEX_ATTRIBUTES).collect(),
			..Default::default()
		};

		let rasterization_state = RasterizationState {
			cull_mode: CullMode::Back,
			..Default::default()
		};

		let rendering_formats = PipelineRenderingCreateInfo {
			color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
			depth_attachment_format: Some(depth_format),
			..Default::default()
		};

		let color_blend_state = ColorBlendState {
			attachments: vec![ColorBlendAttachmentState {
				blend: attachment_blend,
				..Default::default()
			}],
			..Default::default()
		};

		let depth_stencil_state = DepthStencilState {
			depth: Some(DepthState::simple()),
			..Default::default()
		};

		// Create the opaque pass pipeline.
		let opaque_pipeline_info = GraphicsPipelineCreateInfo {
			stages: smallvec::smallvec![
				vs_stage.clone(),
				PipelineShaderStageCreateInfo::new(self.fragment_shader.entry_point("main").unwrap()),
			],
			vertex_input_state: Some(vertex_input_state.clone()),
			input_assembly_state: Some(Default::default()),
			viewport_state: Some(Default::default()),
			rasterization_state: Some(rasterization_state.clone()),
			multisample_state: Some(Default::default()),
			depth_stencil_state: Some(depth_stencil_state),
			color_blend_state: Some(color_blend_state),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(rendering_formats.into()),
			..GraphicsPipelineCreateInfo::layout(pipeline_layout.clone())
		};
		let opaque_pipeline = GraphicsPipeline::new(device.clone(), None, opaque_pipeline_info)?;

		// Create the transparency pass pipeline.
		let oit_pipeline = fs_oit
			.map(|fs| {
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

				let oit_color_blend_state = ColorBlendState {
					attachments: vec![
						ColorBlendAttachmentState {
							// accum
							blend: Some(AttachmentBlend {
								alpha_blend_op: BlendOp::Add,
								..AttachmentBlend::additive()
							}),
							..Default::default()
						},
						ColorBlendAttachmentState {
							// revealage
							blend: Some(AttachmentBlend {
								color_blend_op: BlendOp::Add,
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
					depth_attachment_format: Some(depth_format),
					stencil_attachment_format: Some(depth_format),
					..Default::default()
				};

				let oit_pipeline_info = GraphicsPipelineCreateInfo {
					stages: smallvec::smallvec![vs_stage, PipelineShaderStageCreateInfo::new(fs.entry_point("main").unwrap())],
					vertex_input_state: Some(vertex_input_state),
					input_assembly_state: Some(Default::default()),
					viewport_state: Some(Default::default()),
					rasterization_state: Some(rasterization_state),
					multisample_state: Some(Default::default()),
					depth_stencil_state: Some(oit_depth_stencil_state),
					color_blend_state: Some(oit_color_blend_state),
					dynamic_state: [DynamicState::Viewport].into_iter().collect(),
					subpass: Some(oit_rendering_formats.into()),
					..GraphicsPipelineCreateInfo::layout(pipeline_layout_oit)
				};
				GraphicsPipeline::new(device, None, oit_pipeline_info)
			})
			.transpose()?;

		Ok(MaterialPipelines {
			opaque_pipeline,
			oit_pipeline,
		})
	}
}
