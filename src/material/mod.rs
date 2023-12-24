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
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::device::{Device, DeviceOwned};
use vulkano::format::Format;
use vulkano::pipeline::{
	graphics::{color_blend::AttachmentBlend, input_assembly::PrimitiveTopology},
	GraphicsPipeline
};
use vulkano::shader::ShaderModule;

use crate::render::{texture::Texture, RenderContext};
use crate::EngineError;

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

	fn load_pipeline(&self, vk_dev: Arc<Device>) -> MaterialPipelineConfig;
}

#[derive(Debug)]
pub enum ShaderInput
{
	Color(ColorInput),
	Greyscale(GreyscaleInput),
}
impl ShaderInput
{
	pub fn into_texture(&self, path_prefix: &Path, render_ctx: &mut RenderContext) -> Result<Arc<Texture>, EngineError>
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
	pub fn into_texture(&self, path_prefix: &Path, render_ctx: &mut RenderContext) -> Result<Arc<Texture>, EngineError>
	{
		match self {
			Self::Color(color) => {
				// If the input is a single color, make a 1x1 RGBA texture with just the color.
				Ok(Arc::new(render_ctx.new_texture_from_slice(
					&[*color],
					Format::R32G32B32A32_SFLOAT,
					[1, 1],
					1,
					1,
				)?))
			}
			Self::Texture(tex_path) => render_ctx.get_texture(&path_prefix.join(tex_path)),
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
	pub fn into_texture(&self, path_prefix: &Path, render_ctx: &mut RenderContext) -> Result<Arc<Texture>, EngineError>
	{
		match self {
			Self::Value(value) => {
				// If the input is a single value, make a 2x2 greyscale texture with just the value.
				// (we make it 2x2 here so that it's aligned to 16 bytes)
				Ok(Arc::new(render_ctx.new_texture_from_slice(
					&vec![*value; 4],
					Format::R32_SFLOAT,
					[2, 2],
					1,
					1,
				)?))
			}
			Self::Texture(tex_path) => render_ctx.get_texture(&path_prefix.join(tex_path)),
		}
	}
}

pub enum MaterialTransparencyMode
{
	NoTransparency,
	Blend(AttachmentBlend),
	OIT(Arc<ShaderModule>),
}
impl MaterialTransparencyMode
{
	fn get_blend_or_shader(&self) -> (Option<AttachmentBlend>, Option<Arc<ShaderModule>>)
	{
		match self {
			Self::NoTransparency => (None, None),
			Self::Blend(blend) => (Some(blend.clone()), None),
			Self::OIT(fs_oit) => (None, Some(fs_oit.clone())),
		}
	}
}
pub struct MaterialPipelineConfig
{
	pub primitive_topology: PrimitiveTopology,
	pub vertex_shader: Arc<ShaderModule>,
	pub fragment_shader: Arc<ShaderModule>,
	pub transparency: MaterialTransparencyMode,
}
impl MaterialPipelineConfig
{
	pub fn load_pipeline(
		&self,
		material_textures_set_layout: Arc<DescriptorSetLayout>,
		light_set_layout: Arc<DescriptorSetLayout>,
		transparency_inputs: Arc<DescriptorSetLayout>,
	) -> Result<(Arc<GraphicsPipeline>, Option<Arc<GraphicsPipeline>>), EngineError>
	{
		let vk_dev = material_textures_set_layout.device().clone();

		let (color_blend_state, fs_oit) = self.transparency.get_blend_or_shader();

		let pipeline = crate::render::pipeline::new_for_material(
			vk_dev.clone(),
			self.vertex_shader.clone(),
			self.fragment_shader.clone(),
			color_blend_state,
			self.primitive_topology,
			vec![material_textures_set_layout.clone(), light_set_layout.clone()],
		)?;
		let transparency_pipeline = fs_oit
			.map(|fs| {
				crate::render::pipeline::new_for_material_transparency(
					vk_dev,
					self.vertex_shader.clone(),
					fs,
					self.primitive_topology,
					vec![material_textures_set_layout.clone(), light_set_layout, transparency_inputs],
				)
			})
			.transpose()?;

		Ok((pipeline, transparency_pipeline))
	}
}
