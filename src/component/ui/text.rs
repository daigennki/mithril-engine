/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
use glam::*;
use serde::Deserialize;
use shipyard::{IntoIter, IntoWithId, IntoWorkloadSystem, UniqueViewMut, View, WorkloadSystem};

use crate::component::{ComponentSystems, EntityComponent};
use crate::SystemBundle;

/// UI component meant to render the text to a quad.
#[derive(shipyard::Component, Deserialize, EntityComponent)]
#[track(All)]
pub struct UIText
{
	pub text_str: String,
	pub size: f64,
	pub color: Vec4,
}
impl ComponentSystems for UIText
{
	fn late_update() -> Option<WorkloadSystem>
	{
		Some(update_text.into_workload_system().unwrap())
	}
}
fn update_text(
	mut render_ctx: UniqueViewMut<crate::render::RenderContext>,
	mut canvas: UniqueViewMut<crate::render::ui::Canvas>,
	ui_transforms: View<super::UITransform>,
	ui_texts: View<UIText>,
)
{
	for eid in ui_texts.removed() {
		canvas.cleanup_removed_text(eid);
	}

	if render_ctx.window_resized() {
		let d = render_ctx.window_dimensions();
		canvas.on_screen_resize(d[0], d[1]);

		for (eid, (t, text)) in (&ui_transforms, &ui_texts).iter().with_id() {
			if let Err(e) = canvas.update_text(&mut render_ctx, eid, t, text) {
				log::error!("Failed to update UI text: {}", e);
			}
		}
	} else {
		// Update inserted or modified components.
		for (eid, (t, text)) in (ui_transforms.inserted_or_modified(), &ui_texts).iter().with_id() {
			if let Err(e) = canvas.update_text(&mut render_ctx, eid, t, text) {
				log::error!("Failed to update UI text: {}", e);
			}
		}

		// `Not` is used on `inserted_or_modified` here so that we don't run the updates twice.
		for (eid, (t, text)) in (!ui_transforms.inserted_or_modified(), ui_texts.inserted_or_modified())
			.iter()
			.with_id()
		{
			if let Err(e) = canvas.update_text(&mut render_ctx, eid, t, text) {
				log::error!("Failed to update UI text: {}", e);
			}
		}
	}
}
