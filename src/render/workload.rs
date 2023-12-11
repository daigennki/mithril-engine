/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use glam::*;
use shipyard::{
	iter::{IntoIter, IntoWithId},
	UniqueView, UniqueViewMut, View, Workload,
};
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::format::Format;
use vulkano::pipeline::{graphics::GraphicsPipeline, Pipeline, PipelineBindPoint};

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
	transforms: View<crate::component::Transform>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
	mut light_manager: UniqueViewMut<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let dir_light_extent = light_manager.get_dir_light_shadow().image().extent();
	let viewport_extent = [dir_light_extent[0], dir_light_extent[1]];
	let shadow_pipeline = light_manager.get_shadow_pipeline().clone();
	let shadow_format = Some(light_manager.get_dir_light_shadow().format());

	for layer_projview in light_manager.get_dir_light_projviews() {
		let mut cb = render_ctx.gather_commands(&[], shadow_format, None, viewport_extent)?;

		cb.bind_pipeline_graphics(shadow_pipeline.clone())?;

		for (eid, transform) in transforms.iter().with_id() {
			if mesh_manager.has_opaque_materials(eid) {
				let model_matrix = transform.get_matrix();
				let transform_mat = layer_projview * model_matrix;
				let model_mat3a = Mat3A::from_mat4(model_matrix);

				mesh_manager.draw(
					eid,
					&mut cb,
					shadow_pipeline.layout().clone(),
					transform_mat,
					model_mat3a,
					transform.position,
					false,
					false,
					true,
				)?;
			}
		}

		light_manager.add_dir_light_cb(cb.build()?);
	}

	Ok(())
}

// Draw 3D objects.
// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
fn draw_common(
	command_buffer: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
	camera_manager: &CameraManager,
	transforms: View<crate::component::Transform>,
	mesh_manager: &crate::component::mesh::MeshManager,
	pipeline: &Arc<GraphicsPipeline>,
	transparency_pass: bool,
	base_color_only: bool,
) -> Result<(), GenericEngineError>
{
	let projview = camera_manager.projview();

	for (eid, transform) in transforms.iter().with_id() {
		if (mesh_manager.has_opaque_materials(eid) && !transparency_pass)
			|| (mesh_manager.has_transparency(eid) && transparency_pass)
		{
			let model_matrix = transform.get_matrix();
			let transform_mat = projview * model_matrix;
			let model_mat3a = Mat3A::from_mat4(model_matrix);
			mesh_manager.draw(
				eid,
				command_buffer,
				pipeline.layout().clone(),
				transform_mat,
				model_mat3a,
				transform.position,
				transparency_pass,
				base_color_only,
				false,
			)?;
		}
	}
	Ok(())
}

// Draw opaque 3D objects
fn draw_3d(
	render_ctx: UniqueView<RenderContext>,
	skybox: UniqueView<super::skybox::Skybox>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<crate::component::Transform>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
	light_manager: UniqueView<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let color_formats = [Format::R16G16B16A16_SFLOAT];
	let vp_extent = render_ctx.swapchain_dimensions();
	let light_set = vec![light_manager.get_all_lights_set().clone()];

	let mut cb = render_ctx.gather_commands(&color_formats, Some(super::MAIN_DEPTH_FORMAT), None, vp_extent)?;

	// Draw the skybox. This will effectively clear the color image.
	skybox.draw(&mut cb, camera_manager.sky_projview())?;

	let pbr_pipeline = render_ctx.get_pipeline("PBR").ok_or("PBR pipeline not loaded!")?;

	cb.bind_pipeline_graphics(pbr_pipeline.clone())?
		.push_constants(pbr_pipeline.layout().clone(), 0, camera_manager.projview())?
		.bind_descriptor_sets(PipelineBindPoint::Graphics, pbr_pipeline.layout().clone(), 1, light_set)?;

	draw_common(
		&mut cb,
		&camera_manager,
		transforms,
		&mesh_manager,
		pbr_pipeline,
		false,
		false,
	)?;

	render_ctx.add_cb(cb.build()?);
	Ok(())
}

// Start recording commands for moment-based OIT.
fn draw_3d_transparent_moments(
	render_ctx: UniqueView<RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<crate::component::Transform>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
) -> Result<(), GenericEngineError>
{
	let color_formats = [Format::R32G32B32A32_SFLOAT, Format::R32_SFLOAT, Format::R32_SFLOAT];
	let vp_extent = render_ctx.swapchain_dimensions();
	let mut cb = render_ctx.gather_commands(&color_formats, Some(super::MAIN_DEPTH_FORMAT), None, vp_extent)?;

	// This will bind the pipeline for you, since it doesn't need to do anything
	// specific to materials (it only reads the alpha channel of each texture).
	let pipeline = render_ctx.get_transparency_renderer().get_moments_pipeline();

	cb.bind_pipeline_graphics(pipeline.clone())?;

	draw_common(&mut cb, &camera_manager, transforms, &mesh_manager, pipeline, true, true)?;

	render_ctx
		.get_transparency_renderer()
		.add_transparency_moments_cb(cb.build()?);

	Ok(())
}

// Draw the transparent objects.
fn draw_3d_transparent(
	render_ctx: UniqueView<RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<crate::component::Transform>,
	mesh_manager: UniqueView<crate::component::mesh::MeshManager>,
	light_manager: UniqueView<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let color_formats = [Format::R16G16B16A16_SFLOAT, Format::R8_UNORM];
	let vp_extent = render_ctx.swapchain_dimensions();
	let pipeline = render_ctx
		.get_transparency_pipeline("PBR")
		.ok_or("PBR transparency pipeline not loaded!")?;
	let common_sets = vec![
		light_manager.get_all_lights_set().clone(),
		render_ctx.get_transparency_renderer().get_stage3_inputs().clone(),
	];

	let mut cb = render_ctx.gather_commands(&color_formats, Some(super::MAIN_DEPTH_FORMAT), None, vp_extent)?;

	cb.bind_pipeline_graphics(pipeline.clone())?.bind_descriptor_sets(
		PipelineBindPoint::Graphics,
		pipeline.layout().clone(),
		1,
		common_sets,
	)?;

	draw_common(&mut cb, &camera_manager, transforms, &mesh_manager, pipeline, true, false)?;

	render_ctx.get_transparency_renderer().add_transparency_cb(cb.build()?);

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
	mut canvas: UniqueViewMut<ui::canvas::Canvas>,
	mut light_manager: UniqueViewMut<crate::component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	render_ctx.submit_frame(canvas.take_cb(), light_manager.drain_dir_light_cb())
}
