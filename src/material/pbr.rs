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
use vulkano::device::Device;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::pipeline::{graphics::input_assembly::PrimitiveTopology, layout::PushConstantRange};
use vulkano::shader::ShaderStages;

use super::{ColorInput, /*SingleChannelInput,*/ Material};
use crate::render::{pipeline::PipelineConfig, RenderContext};
use crate::GenericEngineError;

pub mod fs
{
	vulkano_shaders::shader! {
		ty: "fragment",
		path: "src/shaders/pbr.frag.glsl",
	}
}
pub mod fs_oit
{
	vulkano_shaders::shader! {
		ty: "fragment",
		define: [("TRANSPARENCY_PASS", ""),],
		path: "src/shaders/pbr.frag.glsl",
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
	pub fn set_layout_pbr(vk_dev: Arc<Device>) -> Result<Arc<DescriptorSetLayout>, GenericEngineError>
	{
		let sampler_info = SamplerCreateInfo {
			anisotropy: Some(16.0),
			..SamplerCreateInfo::simple_repeat_linear()
		};
		let sampler = Sampler::new(vk_dev.clone(), sampler_info)?;

		let bindings = [
			(
				0,
				DescriptorSetLayoutBinding {
					// binding 0: sampler0
					stages: ShaderStages::FRAGMENT,
					immutable_samplers: vec![sampler],
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
				},
			),
			(
				1,
				DescriptorSetLayoutBinding {
					// binding 1: base_color
					stages: ShaderStages::FRAGMENT,
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
				},
			),
		];
		let layout_info = DescriptorSetLayoutCreateInfo {
			bindings: bindings.into(),
			..Default::default()
		};

		Ok(DescriptorSetLayout::new(vk_dev, layout_info)?)
	}

	pub fn get_pipeline_config(vk_dev: Arc<Device>) -> Result<PipelineConfig, GenericEngineError>
	{
		let pbr_set_layout = Self::set_layout_pbr(vk_dev.clone())?;
		Ok(crate::render::pipeline::PipelineConfig {
			vertex_shader: super::vs_3d_common::load(vk_dev.clone())?,
			fragment_shader: fs::load(vk_dev.clone())?,
			fragment_shader_transparency: Some(fs_oit::load(vk_dev)?),
			attachment_blend: None, // transparency will be handled by transparency renderer
			primitive_topology: PrimitiveTopology::TriangleList,
			set_layouts: vec![pbr_set_layout],
			push_constant_ranges: vec![PushConstantRange {
				// push constant for projviewmodel and transform3 matrix
				stages: ShaderStages::VERTEX,
				offset: 0,
				size: std::mem::size_of::<crate::component::mesh::MeshPushConstant>()
					.try_into()
					.unwrap(),
			}],
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
		render_ctx: &mut RenderContext,
	) -> Result<Vec<WriteDescriptorSet>, GenericEngineError>
	{
		let base_color = self.base_color.into_texture(parent_folder, render_ctx)?;

		let writes = vec![WriteDescriptorSet::image_view(1, base_color.view().clone())];

		Ok(writes)
	}

	fn gen_base_color_descriptor_set_writes(
		&self,
		parent_folder: &Path,
		render_ctx: &mut RenderContext,
	) -> Result<Vec<WriteDescriptorSet>, GenericEngineError>
	{
		let base_color = self.base_color.into_texture(parent_folder, render_ctx)?;

		let writes = vec![WriteDescriptorSet::image_view(1, base_color.view().clone())];

		Ok(writes)
	}

	fn has_transparency(&self) -> bool
	{
		self.transparent
	}
}
