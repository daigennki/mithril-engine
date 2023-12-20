/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use shipyard::{UniqueView, UniqueViewMut, Workload};

use super::RenderContext;
use crate::component::camera::CameraManager;
use crate::component::mesh::PassType;
use crate::component::ui;
use crate::GenericEngineError;

pub fn render() -> Workload
{
	Workload::new("Render")
		.with_try_system(|mut render_ctx: UniqueViewMut<RenderContext>| render_ctx.submit_async_transfers())
		.with_try_system(draw_shadows)
		.with_try_system(draw_3d)
		.with_try_system(draw_3d_oit)
		.with_try_system(draw_ui)
		.with_try_system(submit_frame)
}

// Render shadow maps.
fn draw_shadows(
	render_ctx: UniqueView<RenderContext>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
	mut light_manager: UniqueViewMut<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let dir_light_extent = light_manager.get_dir_light_shadow().image().extent();
	let viewport_extent = [dir_light_extent[0], dir_light_extent[1]];
	let pipeline = light_manager.get_shadow_pipeline().clone();
	let format = light_manager.get_dir_light_shadow().format();

	for layer_projview in light_manager.get_dir_light_projviews() {
		let cb = mesh_manager.draw(
			&render_ctx,
			layer_projview,
			PassType::Shadow {
				pipeline: pipeline.clone(),
				format,
				viewport_extent,
			},
			&[],
		)?;
		light_manager.add_dir_light_cb(cb.unwrap());
	}

	Ok(())
}

// Draw opaque 3D objects
fn draw_3d(
	render_ctx: UniqueView<RenderContext>,
	mut skybox: UniqueViewMut<super::skybox::Skybox>,
	camera_manager: UniqueView<CameraManager>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
	light_manager: UniqueView<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	skybox.draw(&render_ctx, camera_manager.sky_projview())?;

	let common_sets = [light_manager.get_all_lights_set().clone()];
	let cb = mesh_manager.draw(&render_ctx, camera_manager.projview(), PassType::Opaque, &common_sets)?;

	mesh_manager.add_cb(cb.unwrap());
	Ok(())
}

// Draw objects for OIT (order-independent transparency).
fn draw_3d_oit(
	render_ctx: UniqueView<RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
	light_manager: UniqueView<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	// First, collect moments for Moment-based OIT.
	// This will bind the pipeline for you, since it doesn't need to do anything
	// specific to materials (it only reads the alpha channel of each texture).
	let moments_pipeline = render_ctx.get_transparency_renderer().get_moments_pipeline().clone();
	let moments_cb = mesh_manager.draw(
		&render_ctx,
		camera_manager.projview(),
		PassType::TransparencyMoments(moments_pipeline),
		&[],
	)?;
	if let Some(some_moments_cb) = moments_cb {
		render_ctx
			.get_transparency_renderer()
			.add_transparency_moments_cb(some_moments_cb);

		// Now, do the weights pass for OIT.
		let common_sets = [
			light_manager.get_all_lights_set().clone(),
			render_ctx.get_transparency_renderer().get_stage3_inputs().clone(),
		];
		let cb = mesh_manager.draw(&render_ctx, camera_manager.projview(), PassType::Transparency, &common_sets)?;
		render_ctx.get_transparency_renderer().add_transparency_cb(cb.unwrap());
	}

	Ok(())
}

// Draw UI elements.
fn draw_ui(
	render_ctx: UniqueView<RenderContext>,
	mut canvas: UniqueViewMut<ui::canvas::Canvas>,
) -> Result<(), GenericEngineError>
{
	canvas.draw(&render_ctx)
}

fn submit_frame(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut skybox: UniqueViewMut<super::skybox::Skybox>,
	mut mesh_manager: UniqueViewMut<crate::component::mesh::MeshManager>,
	mut canvas: UniqueViewMut<ui::canvas::Canvas>,
	mut light_manager: UniqueViewMut<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let sky_cb = skybox.take_cb().unwrap();
	let mesh_3d_cb = mesh_manager.take_cb().unwrap();
	let ui_cb = canvas.take_cb();
	let dir_light_cb = light_manager.drain_dir_light_cb();

	render_ctx.submit_frame(sky_cb, mesh_3d_cb, ui_cb, dir_light_cb)
}
