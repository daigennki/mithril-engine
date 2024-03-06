/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use serde::Deserialize;
use std::sync::Arc;
use vulkano::device::Device;

use super::{ColorInput, Material, MaterialPipelineConfig, ShaderInput};

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
	//pub roughness: GreyscaleInput,
	//pub specular: GreyscaleInput,
	#[serde(default)]
	pub transparent: bool,
}

#[typetag::deserialize]
impl Material for PBR
{
	fn material_name(&self) -> &'static str
	{
		"PBR"
	}

	fn get_shader_inputs(&self) -> Vec<ShaderInput>
	{
		vec![self.base_color.clone().into()]
	}

	fn has_transparency(&self) -> bool
	{
		self.transparent
	}

	fn load_shaders(&self, vk_dev: Arc<Device>) -> crate::Result<MaterialPipelineConfig>
	{
		Ok(MaterialPipelineConfig {
			vertex_shader: super::vs_3d_common::load(vk_dev.clone())?,
			fragment_shader: fs::load(vk_dev.clone())?,
			fragment_shader_oit: Some(fs_oit::load(vk_dev)?),
		})
	}
}
