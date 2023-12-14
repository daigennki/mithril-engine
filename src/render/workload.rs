/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use shipyard::{
	iter::{IntoIter, IntoWithId},
	UniqueView, UniqueViewMut, View, Workload,
};
use vulkano::format::Format;

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
	let shadow_format = Some(light_manager.get_dir_light_shadow().format());

	for layer_projview in light_manager.get_dir_light_projviews() {
		let cb_builder = render_ctx.gather_commands(&[], shadow_format, None, viewport_extent)?;
		let cb_built = mesh_manager.draw(
			cb_builder,
			layer_projview,
			Some(shadow_pipeline.clone()),
			false,
			false,
			true,
			&[],
		)?;

		light_manager.add_dir_light_cb(cb_built);
	}

	Ok(())
}

// Draw opaque 3D objects
fn draw_3d(
	render_ctx: UniqueView<RenderContext>,
	skybox: UniqueView<super::skybox::Skybox>,
	camera_manager: UniqueView<CameraManager>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
	light_manager: UniqueView<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let color_formats = [Format::R16G16B16A16_SFLOAT];
	let vp_extent = render_ctx.swapchain_dimensions();
	let common_sets = [light_manager.get_all_lights_set().clone()];

	let mut sky_cb = render_ctx.gather_commands(&color_formats, None, None, vp_extent)?;
	skybox.draw(&mut sky_cb, camera_manager.sky_projview())?;
	mesh_manager.add_cb(sky_cb.build()?);

	let cb_builder = render_ctx.gather_commands(&color_formats, Some(super::MAIN_DEPTH_FORMAT), None, vp_extent)?;
	let cb_built = mesh_manager.draw(cb_builder, camera_manager.projview(), None, false, false, false, &common_sets)?;

	mesh_manager.add_cb(cb_built);
	Ok(())
}

// Start recording commands for moment-based OIT.
fn draw_3d_transparent_moments(
	render_ctx: UniqueView<RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
) -> Result<(), GenericEngineError>
{
	let color_formats = [Format::R32G32B32A32_SFLOAT, Format::R32_SFLOAT, Format::R32_SFLOAT];
	let vp_extent = render_ctx.swapchain_dimensions();

	// This will bind the pipeline for you, since it doesn't need to do anything
	// specific to materials (it only reads the alpha channel of each texture).
	let pipeline = render_ctx.get_transparency_renderer().get_moments_pipeline().clone();

	let cb_builder = render_ctx.gather_commands(&color_formats, Some(super::MAIN_DEPTH_FORMAT), None, vp_extent)?;
	let cb_built = mesh_manager.draw(cb_builder, camera_manager.projview(), Some(pipeline), true, true, false, &[])?;

	render_ctx.get_transparency_renderer().add_transparency_moments_cb(cb_built);

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
	let color_formats = [Format::R16G16B16A16_SFLOAT, Format::R8_UNORM];
	let vp_extent = render_ctx.swapchain_dimensions();
	let common_sets = [
		light_manager.get_all_lights_set().clone(),
		render_ctx.get_transparency_renderer().get_stage3_inputs().clone(),
	];

	let cb_builder = render_ctx.gather_commands(&color_formats, Some(super::MAIN_DEPTH_FORMAT), None, vp_extent)?;
	let cb_built = mesh_manager.draw(cb_builder, camera_manager.projview(), None, true, false, false, &common_sets)?;

	render_ctx.get_transparency_renderer().add_transparency_cb(cb_built);

	Ok(())
}

// Draw UI elements.
fn draw_ui(
	render_ctx: UniqueView<RenderContext>,
	mut canvas: UniqueViewMut<ui::canvas::Canvas>,
	ui_transforms: View<ui::UITransform>,
	ui_meshes: View<ui::mesh::Mesh>,
	ui_texts: View<ui::text::UIText>,
) -> Result<(), GenericEngineError>
{
	let vp_extent = render_ctx.swapchain_dimensions();
	let mut cb = render_ctx.gather_commands(&[Format::R16G16B16A16_SFLOAT], None, None, vp_extent)?;

	cb.bind_pipeline_graphics(canvas.get_pipeline().clone())?;

	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	for (eid, (_, ui_mesh)) in (&ui_transforms, &ui_meshes).iter().with_id() {
		// TODO: how do we respect the render order?
		canvas.draw(&mut cb, eid, ui_mesh)?;
	}

	cb.bind_pipeline_graphics(canvas.get_text_pipeline().clone())?;
	for (eid, _) in (&ui_transforms, &ui_texts).iter().with_id() {
		// TODO: how do we respect the render order?
		canvas.draw_text(&mut cb, eid)?;
	}

	canvas.add_cb(cb.build()?);
	Ok(())
}

fn submit_frame(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut mesh_manager: UniqueViewMut<crate::component::mesh::MeshManager>,
	mut canvas: UniqueViewMut<ui::canvas::Canvas>,
	mut light_manager: UniqueViewMut<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	render_ctx.submit_frame(mesh_manager.take_cb(), canvas.take_cb(), light_manager.drain_dir_light_cb())
}
