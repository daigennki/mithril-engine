/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod pbr;

use glam::*;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::format::Format;
use vulkano::image::{ImageDimensions, MipmapsCount};

use crate::render::{texture::Texture, RenderContext};
use crate::GenericEngineError;

/// Trait which allows materials to defer loading using `RenderContext` to after deserialization.
/// This allows each `Material` implementor to define the loading function differently.
pub trait DeferMaterialLoading
{
	/// Finish loading the material's descriptor set.
	fn update_descriptor_set(&mut self, path_to_this: &Path, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>;

	fn get_descriptor_set(&self) -> Option<&Arc<PersistentDescriptorSet>>;

	fn get_base_color(&self) -> Option<Vec4>;
	fn set_base_color(&mut self, color: Vec4, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>;
}

/// A material used by meshes to define shader parameters.
/// Derive from this, then also define a loading function and descriptor set getter by implementing
/// `DeferMaterialLoading` manually.
#[typetag::deserialize]
pub trait Material: Send + Sync + DeferMaterialLoading
{
	fn pipeline_name(&self) -> &'static str;

	fn bind_descriptor_set(
		&self,
		command_buffer: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
	) -> Result<(), GenericEngineError>;

	fn has_transparency(&self) -> bool;
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
				Ok(Arc::new(render_ctx.new_texture_from_iter(
					color.to_array(),
					Format::R32G32B32A32_SFLOAT,
					ImageDimensions::Dim2d {
						width: 1,
						height: 1,
						array_layers: 1,
					},
					MipmapsCount::One,
				)?))
			}
			ColorInput::Texture(tex_path) => render_ctx.get_texture(&path_prefix.join(tex_path)),
		}
	}
}

/// A representation of the possible shader greyscale inputs, like those on the shader nodes in Blender.
#[derive(Deserialize)]
#[serde(untagged)]
enum SingleChannelInput
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
				Ok(Arc::new(render_ctx.new_texture_from_iter(
					[*value],
					Format::R32_SFLOAT,
					ImageDimensions::Dim2d {
						width: 1,
						height: 1,
						array_layers: 1,
					},
					MipmapsCount::One,
				)?))
			}
			SingleChannelInput::Texture(tex_path) => render_ctx.get_texture(&path_prefix.join(tex_path)),
		}
	}
}
