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
use vulkano::descriptor_set::{layout::DescriptorSetLayout, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::pipeline::GraphicsPipeline;

use crate::render::{texture::Texture, RenderContext};
use crate::GenericEngineError;

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
	fn material_name_associated() -> &'static str
		where Self: Sized;

	fn material_name(&self) -> &'static str;

	/// Generate descriptor set writes for creating a descriptor set.
	/// Call this when the user (e.g. a `Mesh` component) is created, and when this material is modified.
	fn gen_descriptor_set_writes(
		&self,
		parent_folder: &Path,
		render_ctx: &mut RenderContext,
	) -> Result<Vec<WriteDescriptorSet>, GenericEngineError>;

	fn gen_base_color_descriptor_set_writes(
		&self,
		parent_folder: &Path,
		render_ctx: &mut RenderContext,
	) -> Result<Vec<WriteDescriptorSet>, GenericEngineError>;

	fn has_transparency(&self) -> bool;

	fn load_pipeline(light_set_layout: Arc<DescriptorSetLayout>, transparency_inputs: Arc<DescriptorSetLayout>)
		-> Result<(Arc<GraphicsPipeline>, Option<Arc<GraphicsPipeline>>, Arc<DescriptorSetLayout>), GenericEngineError>
		where Self: Sized;
}

/// A representation of the possible shader color inputs, like those on the shader nodes in Blender.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum ColorInput
{
	/// Single color value. This is in linear color space (*not* gamma corrected) for consistency with Blender's RGB color
	/// picker (https://docs.blender.org/manual/en/latest/interface/controls/templates/color_picker.html).
	Color(Vec4),

	/// A texture image file.
	Texture(PathBuf),
}
impl ColorInput
{
	pub fn into_texture(&self, path_prefix: &Path, render_ctx: &mut RenderContext) -> Result<Arc<Texture>, GenericEngineError>
	{
		match self {
			ColorInput::Color(color) => {
				// If the input is a single color, make a 1x1 RGBA texture with just the color.
				Ok(Arc::new(render_ctx.new_texture_from_slice(
					&[*color],
					Format::R32G32B32A32_SFLOAT,
					[1, 1],
					1,
					1,
				)?))
			}
			ColorInput::Texture(tex_path) => render_ctx.get_texture(&path_prefix.join(tex_path)),
		}
	}
}

/*/// A representation of the possible shader greyscale inputs, like those on the shader nodes in Blender.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum SingleChannelInput
{
	Value(f32),
	Texture(PathBuf),
}
impl SingleChannelInput
{
	pub fn into_texture(&self, path_prefix: &Path, render_ctx: &mut RenderContext) -> Result<Arc<Texture>, GenericEngineError>
	{
		match self {
			SingleChannelInput::Value(value) => {
				// If the input is a single value, make a 1x1 greyscale texture with just the value.
				Ok(Arc::new(render_ctx.new_texture_from_iter([*value], Format::R32_SFLOAT, [ 1, 1 ], 1)?))
			}
			SingleChannelInput::Texture(tex_path) => render_ctx.get_texture(&path_prefix.join(tex_path)),
		}
	}
}*/
