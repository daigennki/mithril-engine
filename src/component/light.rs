/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use serde::Deserialize;
use shipyard::{IntoIter, IntoWorkloadSystem, UniqueViewMut, View, WorkloadSystem};
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::descriptor_set::{
	layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
	PersistentDescriptorSet, WriteDescriptorSet
};
use vulkano::device::DeviceOwned;
use vulkano::shader::ShaderStages;

use crate::{render::RenderContext, GenericEngineError};
use super::{EntityComponent, WantsSystemAdded, Transform};

/// These are various components that represent light sources in the world.
/// 

#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct DirectionalLight
{
	pub color: Vec3,
	pub intensity: f32,
}
impl WantsSystemAdded for DirectionalLight
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		Some(update_directional_light.into_workload_system().unwrap())
	}
}
fn update_directional_light(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut light_manager: UniqueViewMut<LightManager>,
	dir_lights: View<DirectionalLight>,
	transforms: View<Transform>,
)
{
	// Only use the first directional light found, 
	// but also let insertions or modifications that occur later overwrite it;
	// there should really be only one of these in the world anyways.
	if let Some((dl, t)) = (dir_lights.inserted_or_modified(), transforms.inserted_or_modified()).iter().next() {
		if let Err(e) = light_manager.update_dir_light(&mut render_ctx, dl, t) {
			log::error!("Failed to update `DirectionalLight` GPU buffer: {}", e);
		}
	}
}

/*#[derive(shipyard::Component, Deserialze, EntityComponent)]
pub struct PointLight
{
	pub color: Vec3,
	pub intensity: f32,
}

#[derive(shipyard::Component, Deserialze, EntityComponent)]
pub struct SpotLight
{
	pub color: Vec3,
	pub intensity: f32,
}*/

#[derive(Clone, Copy, bytemuck::AnyBitPattern)]
#[repr(C)]
struct DirLightData
{
	direction: Vec3A,
	color_intensity: Vec4, // RGB is color, A is intensity
}
/*struct PointLightData
{
	pub position: Vec3A,
	pub color_intensity: Vec4,
}
struct SpotLightData
{
	pub position: Vec3A,
	pub direction: Vec3A,
	pub color_intensity: Vec4,
}*/

#[derive(shipyard::Unique)]
pub struct LightManager
{
	dir_light_buf: Subbuffer<[DirLightData]>,
	/*point_light_buf: Arc<Subbuffer<[PointLightData]>>,
	spot_light_buf: Arc<Subbuffer<[SpotLightData]>>,*/

	all_lights_set: Arc<PersistentDescriptorSet>,
}
impl LightManager
{
	pub fn new(render_ctx: &mut RenderContext) -> Result<Self, GenericEngineError>
	{
		let dir_light_data = DirLightData {
			direction: Vec3A::NEG_Z,
			color_intensity: Vec4::ONE,
		};

		let dir_light_buf = render_ctx.new_buffer([dir_light_data], BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST)?;

		let device = render_ctx.descriptor_set_allocator().device().clone();
		let set_layout_info = DescriptorSetLayoutCreateInfo {
			bindings: [
				(0, DescriptorSetLayoutBinding { // binding 0: directional light buffer 
					stages: ShaderStages::FRAGMENT,
					..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
				}),
			].into(),
			..Default::default()
		};
		let set_layout = DescriptorSetLayout::new(device.clone(), set_layout_info)?;

		let all_lights_set = PersistentDescriptorSet::new(
			render_ctx.descriptor_set_allocator(),
			set_layout,
			[WriteDescriptorSet::buffer(0, dir_light_buf.clone())],
			[]
		)?;

		Ok(LightManager {
			dir_light_buf,
			all_lights_set,
		})
	}

	pub fn update_dir_light(&mut self, render_ctx: &mut RenderContext, light: &DirectionalLight, transform: &Transform)
		-> Result<(), GenericEngineError>
	{
		let dir_light_data = DirLightData {
			direction: transform.rotation_quat() * Vec3A::Z,
			color_intensity: light.color.extend(light.intensity),
		};
		render_ctx.copy_to_buffer([dir_light_data], self.dir_light_buf.clone())?;
		Ok(())
	}

	pub fn get_all_lights_set(&self) -> &Arc<PersistentDescriptorSet>
	{
		&self.all_lights_set
	}
}

