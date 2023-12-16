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
	layout::{
		DescriptorBindingFlags,DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType
	},
	WriteDescriptorSet,
};
use vulkano::device::DeviceOwned;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::pipeline::{graphics::input_assembly::PrimitiveTopology, GraphicsPipeline};
use vulkano::shader::ShaderStages;

use super::{ColorInput, /*SingleChannelInput,*/ Material};
use crate::render::RenderContext;
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

#[typetag::deserialize]
impl Material for PBR
{
	fn material_name_associated() -> &'static str
	{
		"PBR"
	}
	fn material_name(&self) -> &'static str
	{
		Self::material_name_associated()
	}

	fn gen_descriptor_set_write(
		&self,
		parent_folder: &Path,
		render_ctx: &mut RenderContext,
	) -> Result<WriteDescriptorSet, GenericEngineError>
	{
		let base_color = self.base_color.into_texture(parent_folder, render_ctx)?;

		let image_views = [base_color.view().clone()];

		Ok(WriteDescriptorSet::image_view_array(1, 0, image_views))
	}

	fn gen_base_color_descriptor_set_write(
		&self,
		parent_folder: &Path,
		render_ctx: &mut RenderContext,
	) -> Result<WriteDescriptorSet, GenericEngineError>
	{
		let base_color = self.base_color.into_texture(parent_folder, render_ctx)?;

		Ok(WriteDescriptorSet::image_view(1, base_color.view().clone()))
	}

	fn has_transparency(&self) -> bool
	{
		self.transparent
	}

	fn load_pipeline_associated(
		light_set_layout: Arc<DescriptorSetLayout>,
		transparency_inputs: Arc<DescriptorSetLayout>,
	) -> Result<(Arc<GraphicsPipeline>, Option<Arc<GraphicsPipeline>>, Arc<DescriptorSetLayout>), GenericEngineError>
	{
		let vk_dev = transparency_inputs.device().clone();

		let sampler_info = SamplerCreateInfo {
			anisotropy: Some(16.0),
			..SamplerCreateInfo::simple_repeat_linear()
		};
		let sampler = Sampler::new(vk_dev.clone(), sampler_info)?;
		let bindings = [
			DescriptorSetLayoutBinding {
				// binding 0: sampler0
				stages: ShaderStages::FRAGMENT,
				immutable_samplers: vec![sampler],
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
			},
			DescriptorSetLayoutBinding {
				// binding 1: textures
				binding_flags: DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
				descriptor_count: 32,
				stages: ShaderStages::FRAGMENT,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
			},
		];
		let layout_info = DescriptorSetLayoutCreateInfo {
			bindings: (0..).zip(bindings).collect(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(vk_dev.clone(), layout_info)?;

		let vertex_shader = super::vs_3d_common::load(vk_dev.clone())?;
		let pipeline = crate::render::pipeline::new_for_material(
			vk_dev.clone(),
			vertex_shader.clone(),
			fs::load(vk_dev.clone())?,
			None,
			PrimitiveTopology::TriangleList,
			vec![set_layout.clone(), light_set_layout.clone()],
		)?;
		let transparency_pipeline = crate::render::pipeline::new_for_material_transparency(
			vk_dev.clone(),
			vertex_shader,
			fs_oit::load(vk_dev)?,
			PrimitiveTopology::TriangleList,
			vec![set_layout.clone(), light_set_layout, transparency_inputs],
		)?;

		Ok((pipeline, Some(transparency_pipeline), set_layout))
	}

	fn load_pipeline(
		&self,
		light_set_layout: Arc<DescriptorSetLayout>,
		transparency_inputs: Arc<DescriptorSetLayout>,
	) -> Result<(Arc<GraphicsPipeline>, Option<Arc<GraphicsPipeline>>, Arc<DescriptorSetLayout>), GenericEngineError>
	{
		Self::load_pipeline_associated(light_set_layout, transparency_inputs)
	}
}
