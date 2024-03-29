/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use serde::Deserialize;
use shipyard::{IntoIter, IntoWithId, IntoWorkloadSystem, UniqueViewMut, View, WorkloadSystem};
use std::path::PathBuf;

use crate::component::{ComponentSystems, EntityComponent, SystemBundle};
use crate::render::ui::Canvas;

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(untagged)]
pub enum MeshType
{
	#[default]
	Quad,
	Frame(u32), // `u32` is border width in logical pixels
}

/// UI component that renders to a mesh, such as a quad, or a background frame mesh.
#[derive(Default, Deserialize, EntityComponent, shipyard::Component)]
#[track(All)]
pub struct UIMesh
{
	#[serde(default)]
	pub mesh_type: MeshType,

	pub image_path: PathBuf,
}
impl ComponentSystems for UIMesh
{
	fn late_update() -> Option<WorkloadSystem>
	{
		Some(update_mesh.into_workload_system().unwrap())
	}
}
fn update_mesh(
	mut render_ctx: UniqueViewMut<crate::render::RenderContext>,
	mut canvas: UniqueViewMut<Canvas>,
	ui_transforms: View<super::UITransform>,
	ui_meshes: View<UIMesh>,
)
{
	for eid in ui_meshes.removed() {
		canvas.cleanup_removed_mesh(eid);
	}

	if render_ctx.window_resized() {
		let d = render_ctx.window_dimensions();
		canvas.on_screen_resize(d[0], d[1]);

		for (eid, (t, mesh)) in (&ui_transforms, &ui_meshes).iter().with_id() {
			if let Err(e) = canvas.update_mesh(&mut render_ctx, eid, t, mesh) {
				log::error!("Failed to update UI mesh: {}", e);
			}
		}
	} else {
		for (eid, (t, mesh)) in (ui_transforms.inserted_or_modified(), &ui_meshes).iter().with_id() {
			if let Err(e) = canvas.update_mesh(&mut render_ctx, eid, t, mesh) {
				log::error!("Failed to update UI mesh: {}", e);
			}
		}

		// `Not` is used on `inserted_or_modified` here so that we don't run the updates twice.
		for (eid, (t, mesh)) in (!ui_transforms.inserted_or_modified(), ui_meshes.inserted_or_modified())
			.iter()
			.with_id()
		{
			if let Err(e) = canvas.update_mesh(&mut render_ctx, eid, t, mesh) {
				log::error!("Failed to update UI mesh: {}", e);
			}
		}
	}
}
