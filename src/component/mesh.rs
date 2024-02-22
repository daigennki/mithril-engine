/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use glam::*;
use serde::Deserialize;
use shipyard::{IntoIter, IntoWithId, IntoWorkloadSystem, UniqueViewMut, View, WorkloadSystem};
use std::path::PathBuf;

use crate::component::{EntityComponent, ComponentSystems};
use crate::render::{model::MeshManager, RenderContext};
use crate::SystemBundle;

#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct Mesh
{
	pub model_path: PathBuf,

	#[serde(default)]
	pub material_variant: Option<String>,
}
impl ComponentSystems for Mesh
{
	fn update() -> Option<WorkloadSystem>
	{
		None
	}
	fn late_update() -> Option<WorkloadSystem>
	{
		Some(update_meshes.into_workload_system().unwrap())
	}
}
fn update_meshes(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut mesh_manager: UniqueViewMut<MeshManager>,
	meshes: View<Mesh>,
	transforms: View<super::Transform>,
)
{
	for eid in meshes.removed() {
		mesh_manager.cleanup_removed(eid);
	}

	for (eid, mesh) in meshes.inserted().iter().with_id() {
		if let Err(e) = mesh_manager.load(&mut render_ctx, eid, mesh) {
			log::error!("Failed to run `MeshManager::load`: {}", e);
		}
	}

	for (eid, (t, _)) in (transforms.inserted_or_modified(), &meshes).iter().with_id() {
		mesh_manager.set_affine(eid, t.get_affine());
	}
}
