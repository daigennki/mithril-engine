/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod pbr;

use std::path::{ Path, PathBuf };
use std::sync::Arc;
use glam::*;
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::format::Format;
use vulkano::image::{ ImageDimensions, MipmapsCount };
use serde::Deserialize;
use crate::render::{ RenderContext, command_buffer::CommandBuffer, texture::Texture };
use crate::GenericEngineError;
use mithrilengine_derive::Material;

/// Trait which allows materials to defer loading using `RenderContext` to after deserialization.
/// This allows each `Material` implementor to define the loading function differently.
pub trait DeferMaterialLoading
{
	/// Finish loading the material's descriptor set.
	fn update_descriptor_set(&mut self, path_to_this: &Path, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>;

	fn get_descriptor_set(&self) -> Option<&Arc<PersistentDescriptorSet>>;
}

/// A material used by meshes to define shader parameters.
/// Derive from this using `#[derive(Material)]`, then also define a loading function and descriptor set getter by implementing
/// `DeferMaterialLoading` manually.
#[typetag::deserialize]
pub trait Material: Send + Sync + DeferMaterialLoading
{
	fn pipeline_name(&self) -> &'static str;

	fn bind_descriptor_set(&self, command_buffer: &mut CommandBuffer<SecondaryAutoCommandBuffer>) 
		-> Result<(), GenericEngineError>;
}

/// A representation of the possible shader color inputs, like those on the shader nodes in Blender.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum ColorInput
{
	Color(Vec4),
	Texture(PathBuf)
}
impl ColorInput
{
	pub fn into_texture(&self, path_prefix: &Path, render_ctx: &mut RenderContext) -> Result<Texture, GenericEngineError>
	{
		match self {
			ColorInput::Color(color) => {
				// If the input is a single color, make a 1x1 RGBA texture with just the color.
				// We convert it to sRGB first though, since there's no f32 format for sRGB.
				let linear_color = [
					srgb_to_linear(color.x),
					srgb_to_linear(color.y),
					srgb_to_linear(color.z),
					color.w
				];
				render_ctx.new_texture_from_iter(
					linear_color, 
					Format::R32G32B32A32_SFLOAT, 
					ImageDimensions::Dim2d{ width: 1, height: 1, array_layers: 1 },
					MipmapsCount::One
				)
			},
			ColorInput::Texture(tex_path) => render_ctx.new_texture(&path_prefix.join(tex_path))
		}
	}
}

/// A representation of the possible shader greyscale inputs, like those on the shader nodes in Blender.
#[derive(Deserialize)]
#[serde(untagged)]
enum SingleChannelInput
{
	Value(f32),
	Texture(PathBuf)
}
impl SingleChannelInput
{
	pub fn into_texture(&self, path_prefix: &Path, render_ctx: &mut RenderContext) -> Result<Texture, GenericEngineError>
	{
		match self {
			SingleChannelInput::Value(value) => {
				// If the input is a single value, make a 1x1 greyscale texture with just the value.
				render_ctx.new_texture_from_iter(
					[*value], 
					Format::R32_SFLOAT, 
					ImageDimensions::Dim2d{ width: 1, height: 1, array_layers: 1 },
					MipmapsCount::One
				)
			},
			SingleChannelInput::Texture(tex_path) => render_ctx.new_texture(&path_prefix.join(tex_path))
		}
	}
}

/// Convert non-linear sRGB color into linear sRGB.
fn srgb_to_linear(c: f32) -> f32
{
	if c <= 0.04045 {
		c / 12.92
	} else {
		((c + 0.055) / 1.055).powf(2.4)
	}
}

