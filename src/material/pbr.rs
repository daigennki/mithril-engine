/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use super::{BlendMode, ColorInput, Material, MaterialPipelineConfig, ShaderInput};
use serde::Deserialize;

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
	pub blend_mode: BlendMode,
}

#[typetag::deserialize]
impl Material for PBR
{
	fn name(&self) -> &'static str
	{
		"PBR"
	}

	fn get_shader_inputs(&self) -> Vec<ShaderInput>
	{
		vec![self.base_color.clone().into()]
	}

	fn blend_mode(&self) -> BlendMode
	{
		self.blend_mode
	}
}
inventory::submit! {
	MaterialPipelineConfig {
		type_id: &std::any::TypeId::of::<PBR>,
		vertex_shader: &super::vs_3d_common::load,
		fragment_shader: &fs::load,
		fragment_shader_oit: Some(&fs_oit::load),
	}
}

impl From<&gltf::Material<'_>> for PBR
{
	fn from(material: &gltf::Material) -> Self
	{
		let blend_mode = match material.alpha_mode() {
			// TODO: implement alpha testing
			gltf::material::AlphaMode::Opaque | gltf::material::AlphaMode::Mask => BlendMode::Opaque,
			gltf::material::AlphaMode::Blend => BlendMode::AlphaBlend,
		};

		Self {
			base_color: ColorInput::Color(material.pbr_metallic_roughness().base_color_factor().into()),
			blend_mode,
		}
	}
}
