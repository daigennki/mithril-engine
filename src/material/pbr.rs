/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::device::DeviceOwned;
use vulkano::image::view::ImageView;
use vulkano::pipeline::{graphics::input_assembly::PrimitiveTopology, GraphicsPipeline};

use super::{ColorInput, /*SingleChannelInput,*/ Material};
use crate::render::RenderContext;
use crate::EngineError;

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
	) -> Result<Vec<Arc<ImageView>>, EngineError>
	{
		let base_color = self.base_color.into_texture(parent_folder, render_ctx)?;

		let image_views = vec![base_color.view().clone()];
		Ok(image_views)
	}

	fn has_transparency(&self) -> bool
	{
		self.transparent
	}

	fn load_pipeline_associated(
		material_textures_set_layout: Arc<DescriptorSetLayout>,
		light_set_layout: Arc<DescriptorSetLayout>,
		transparency_inputs: Arc<DescriptorSetLayout>,
	) -> Result<(Arc<GraphicsPipeline>, Option<Arc<GraphicsPipeline>>), EngineError>
	{
		let vk_dev = transparency_inputs.device().clone();

		let vertex_shader = super::vs_3d_common::load(vk_dev.clone()).unwrap();
		let pipeline = crate::render::pipeline::new_for_material(
			vk_dev.clone(),
			vertex_shader.clone(),
			fs::load(vk_dev.clone()).unwrap(),
			None,
			PrimitiveTopology::TriangleList,
			vec![material_textures_set_layout.clone(), light_set_layout.clone()],
		)?;
		let transparency_pipeline = crate::render::pipeline::new_for_material_transparency(
			vk_dev.clone(),
			vertex_shader,
			fs_oit::load(vk_dev).unwrap(),
			PrimitiveTopology::TriangleList,
			vec![material_textures_set_layout.clone(), light_set_layout, transparency_inputs],
		)?;

		Ok((pipeline, Some(transparency_pipeline)))
	}

	fn load_pipeline(
		&self,
		material_textures_set_layout: Arc<DescriptorSetLayout>,
		light_set_layout: Arc<DescriptorSetLayout>,
		transparency_inputs: Arc<DescriptorSetLayout>,
	) -> Result<(Arc<GraphicsPipeline>, Option<Arc<GraphicsPipeline>>), EngineError>
	{
		Self::load_pipeline_associated(material_textures_set_layout, light_set_layout, transparency_inputs)
	}
}
