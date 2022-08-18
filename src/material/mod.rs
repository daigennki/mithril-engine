/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod pbr;

use std::path::PathBuf;
use glam::*;
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use serde::Deserialize;
use crate::render::{ RenderContext, command_buffer::CommandBuffer };
use crate::GenericEngineError;

#[typetag::deserialize]
pub trait Material
{
	fn pipeline_name(&self) -> &'static str;

	fn update_descriptor_set(&mut self, render_ctx: &mut RenderContext) -> Result<(), GenericEngineError>;

	fn bind_descriptor_set(&self, command_buffer: &mut CommandBuffer<SecondaryAutoCommandBuffer>) -> Result<(), GenericEngineError>;
}

/// A representation of the possible shader color inputs, like those on the shader nodes in Blender.
#[derive(Deserialize)]
enum ColorInput
{
	Color(Vec4),
	Texture(String)
}

/// A representation of the possible shader greyscale inputs, like those on the shader nodes in Blender.
#[derive(Deserialize)]
enum SingleChannelInput
{
	Value(f32),
	Texture(String)
}

