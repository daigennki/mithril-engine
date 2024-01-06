/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use shipyard::{UniqueView, UniqueViewMut, Workload};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};

use super::RenderContext;
use crate::component::camera::CameraManager;
use crate::component::mesh::PassType;
use crate::component::ui;
use crate::render::lighting::LightManager;

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
	mut light_manager: UniqueViewMut<LightManager>,
) -> crate::Result<()>
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
	camera_manager: UniqueView<CameraManager>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
	light_manager: UniqueView<LightManager>,
) -> crate::Result<()>
{
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
	light_manager: UniqueView<LightManager>,
) -> crate::Result<()>
{
	// First, collect moments for Moment-based OIT.
	// This will bind the pipeline for you, since it doesn't need to do anything
	// specific to materials (it only reads the alpha channel of each texture).
	let moments_pipeline = render_ctx.transparency_renderer.get_moments_pipeline().clone();
	let moments_cb = mesh_manager.draw(
		&render_ctx,
		camera_manager.projview(),
		PassType::TransparencyMoments(moments_pipeline),
		&[],
	)?;
	if let Some(some_moments_cb) = moments_cb {
		render_ctx.transparency_renderer.add_transparency_moments_cb(some_moments_cb);

		// Now, do the weights pass for OIT.
		let common_sets = [
			light_manager.get_all_lights_set().clone(),
			render_ctx.transparency_renderer.get_stage3_inputs().clone(),
		];
		let cb = mesh_manager.draw(&render_ctx, camera_manager.projview(), PassType::Transparency, &common_sets)?;
		render_ctx.transparency_renderer.add_transparency_cb(cb.unwrap());
	}

	Ok(())
}

// Draw UI elements.
fn draw_ui(render_ctx: UniqueView<RenderContext>, mut canvas: UniqueViewMut<ui::canvas::Canvas>) -> crate::Result<()>
{
	canvas.draw(&render_ctx)
}

/// Submit all the command buffers for this frame to actually render them to the image.
fn submit_frame(
	mut render_ctx: UniqueViewMut<RenderContext>,
	mut skybox: UniqueViewMut<super::skybox::Skybox>,
	mut mesh_manager: UniqueViewMut<crate::component::mesh::MeshManager>,
	mut canvas: UniqueViewMut<ui::canvas::Canvas>,
	mut light_manager: UniqueViewMut<LightManager>,
	camera_manager: UniqueView<CameraManager>,
) -> crate::Result<()>
{
	let mut primary_cb_builder = AutoCommandBufferBuilder::primary(
		&render_ctx.command_buffer_allocator,
		render_ctx.graphics_queue_family_index(),
		CommandBufferUsage::OneTimeSubmit,
	)?;

	render_ctx
		.transfer_manager
		.add_synchronous_transfer_commands(&mut primary_cb_builder)?;

	// Sometimes no image may be returned because the image is out of date or the window is minimized,
	// in which case, don't present.
	if let Some(swapchain_image_num) = render_ctx.swapchain.get_next_image()? {
		if render_ctx.window_resized() {
			render_ctx.resize_everything_else()?;
		}

		let main_render_target = &render_ctx.main_render_target;
		let color_image = main_render_target.color_image().clone();
		let depth_image = main_render_target.depth_image().clone();

		// shadows
		light_manager.execute_shadow_rendering(&mut primary_cb_builder)?;

		// skybox (effectively clears the image)
		skybox.draw(&mut primary_cb_builder, color_image.clone(), camera_manager.sky_projview())?;

		// 3D
		mesh_manager.execute_rendering(&mut primary_cb_builder, color_image.clone(), depth_image.clone())?;

		// 3D OIT
		render_ctx
			.transparency_renderer
			.process_transparency(&mut primary_cb_builder, color_image.clone(), depth_image)?;

		// UI
		canvas.execute_rendering(&mut primary_cb_builder, color_image)?;

		// blit the image to the swapchain image, converting it to the swapchain's color space if necessary
		main_render_target.blit_to_swapchain(&mut primary_cb_builder, swapchain_image_num)?;
	}

	// submit the built command buffer, presenting it if possible
	render_ctx.submit_primary(primary_cb_builder.build()?)
}
