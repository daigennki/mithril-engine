/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::pipeline::{graphics::input_assembly::PrimitiveTopology, layout::PushConstantRange};
use vulkano::shader::ShaderStages;

use super::{ColorInput, /*SingleChannelInput,*/ Material};
use crate::render::{RenderContext, pipeline::PipelineConfig};
use crate::GenericEngineError;

pub mod fs {
	vulkano_shaders::shader! {
		ty: "fragment",
		bytes: "shaders/pbr.frag.spv",
	}
}
pub mod fs_oit {
	vulkano_shaders::shader! {
		ty: "fragment",
		bytes: "shaders/pbr_mboit_weights.frag.spv",
	}
}

/// The standard PBR (Physically Based Rendering) material.
#[derive(Deserialize)]
pub struct PBR
{
	pub base_color: ColorInput,

	// TODO: roughness and specular textures
	//pub roughness: SingleChannelInput,

	//pub specular: SingleChannelInput,

	#[serde(default)]
	pub transparent: bool,
}
impl PBR
{
	pub fn set_layout_pbr(render_ctx: &RenderContext) -> Result<Arc<DescriptorSetLayout>, GenericEngineError>
	{
		let bindings = [
			(0, DescriptorSetLayoutBinding { // binding 0: sampler0
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![ render_ctx.get_default_sampler().clone() ],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
			}),
			(1, DescriptorSetLayoutBinding { // binding 1: base_color
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
			}),
		];
		let layout_info = DescriptorSetLayoutCreateInfo {
			bindings: bindings.into(),
			..Default::default()
		};
		let vk_dev = render_ctx.get_default_sampler().device().clone();
		Ok(DescriptorSetLayout::new(vk_dev.clone(), layout_info)?)
	}

	pub fn get_pipeline_config(render_ctx: &RenderContext) -> Result<PipelineConfig, GenericEngineError>
	{
		let vk_dev = render_ctx.get_default_sampler().device().clone();
		let pbr_set_layout = Self::set_layout_pbr(render_ctx)?;
		Ok(crate::render::pipeline::PipelineConfig {
			vertex_shader: super::vs_3d_common::load(vk_dev.clone())?,
			fragment_shader: fs::load(vk_dev.clone())?,
			fragment_shader_transparency: Some(fs_oit::load(vk_dev.clone())?),
			attachment_blend: None, // transparency will be handled by transparency renderer
			primitive_topology: PrimitiveTopology::TriangleList,
			depth_processing: true,
			set_layouts: vec![ pbr_set_layout ],
			push_constant_ranges: vec![
				PushConstantRange { // push constant for projviewmodel and transform3 matrix
					stages: ShaderStages::VERTEX,
					offset: 0,
					size: (std::mem::size_of::<Mat4>() * 2).try_into().unwrap(),
				}
			],
		})
	}
}

#[typetag::deserialize]
impl Material for PBR
{
	fn material_name(&self) -> &'static str
	{
		"PBR"
	}

	fn gen_descriptor_set_writes(
		&self,
		parent_folder: &Path,
		render_ctx: &mut RenderContext
	) -> Result<Vec<WriteDescriptorSet>, GenericEngineError>
	{
		let base_color = self.base_color.into_texture(parent_folder, render_ctx)?;

		let writes = vec![ WriteDescriptorSet::image_view(1, base_color.view()) ];

		Ok(writes)
	}

	fn set_layout(&self, render_ctx: &RenderContext) -> Result<Arc<DescriptorSetLayout>, GenericEngineError>
	{
		Self::set_layout_pbr(render_ctx)
	}

	fn has_transparency(&self) -> bool
	{
		self.transparent
	}
}

