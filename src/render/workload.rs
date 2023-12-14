/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use shipyard::{UniqueView, UniqueViewMut, Workload};

use super::RenderContext;
use crate::component::camera::CameraManager;
use crate::component::ui;
use crate::GenericEngineError;

pub fn render() -> Workload
{
	Workload::new("Render")
		.with_try_system(|mut render_ctx: UniqueViewMut<RenderContext>| render_ctx.submit_async_transfers())
		.with_try_system(draw_shadows)
		.with_try_system(draw_3d)
		.with_try_system(draw_3d_transparent_moments)
		.with_try_system(draw_3d_transparent)
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
	let shadow_pipeline = light_manager.get_shadow_pipeline().clone();
	let shadow_format = light_manager.get_dir_light_shadow().format();

	for layer_projview in light_manager.get_dir_light_projviews() {
		let cb = mesh_manager.draw(
			&render_ctx,
			layer_projview,
			Some(shadow_pipeline.clone()),
			false,
			Some((shadow_format, viewport_extent)),
			&[],
		)?;
		light_manager.add_dir_light_cb(cb);
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
	let cb = mesh_manager.draw(&render_ctx, camera_manager.projview(), None, false, None, &common_sets)?;

	mesh_manager.add_cb(cb);
	Ok(())
}

// Start recording commands for moment-based OIT.
fn draw_3d_transparent_moments(
	render_ctx: UniqueView<RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
) -> Result<(), GenericEngineError>
{
	// This will bind the pipeline for you, since it doesn't need to do anything
	// specific to materials (it only reads the alpha channel of each texture).
	let pipeline = render_ctx.get_transparency_renderer().get_moments_pipeline().clone();

	let cb = mesh_manager.draw(&render_ctx, camera_manager.projview(), Some(pipeline), true, None, &[])?;

	render_ctx.get_transparency_renderer().add_transparency_moments_cb(cb);

	Ok(())
}

// Draw the transparent objects.
fn draw_3d_transparent(
	render_ctx: UniqueView<RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
	light_manager: UniqueView<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let common_sets = [
		light_manager.get_all_lights_set().clone(),
		render_ctx.get_transparency_renderer().get_stage3_inputs().clone(),
	];

	let cb = mesh_manager.draw(&render_ctx, camera_manager.projview(), None, true, None, &common_sets)?;

	render_ctx.get_transparency_renderer().add_transparency_cb(cb);

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
