/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod component;
pub mod material;
pub mod render;

//#[cfg(feature = "egui")]
//mod egui_renderer;

use glam::*;
use serde::Deserialize;
use shipyard::{iter::{IntoIter, IntoWithId}, UniqueView, UniqueViewMut, View, Workload, World};
use simplelog::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::DeviceOwned;
use vulkano::format::Format;
use vulkano::pipeline::{graphics::GraphicsPipeline, Pipeline, PipelineBindPoint};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit_input_helper::WinitInputHelper;

use component::camera::{CameraFov, CameraManager};
use component::ui;
use component::ui::canvas::Canvas;
use render::RenderContext;

//#[cfg(feature = "egui")]
//use egui_renderer::EguiRenderer;

type GenericEngineError = Box<dyn std::error::Error + Send + Sync>;

/// Run the game. This should be called from your `main.rs`.
/// `org_name` and `game_name` will be used for the data directory.
/// `game_name` will also be used for the window title.
/// `start_map` is the first map (level/world) to be loaded.
pub fn run_game(org_name: &str, game_name: &str, start_map: &str)
{
	let event_loop = match winit::event_loop::EventLoop::new() {
		Ok(el) => el,
		Err(e) => {
			log_error(Box::new(e));
			return
		}
	};

	event_loop.set_control_flow(ControlFlow::Poll);

	let mut world = match init_world(org_name, game_name, start_map, &event_loop) {
		Ok(w) => w,
		Err(e) => {
			log_error(e);
			return;
		}
	};

	if let Err(e) = event_loop.run(move |mut event, window_target| {
		match handle_event(&mut world, &mut event) {
			Ok(true) => window_target.exit(),
			Ok(false) => (),
			Err(e) => {
				log_error(e);
				window_target.exit();
			}
		}
	}) {
		log_error(Box::new(e));
	}
}

#[derive(shipyard::Unique)]
pub struct InputHelperWrapper
{
	pub inner: WinitInputHelper,
}

fn init_world(org_name: &str, game_name: &str, start_map: &str, event_loop: &winit::event_loop::EventLoop<()>)
	-> Result<World, GenericEngineError>
{
	setup_log(org_name, game_name)?;

	let mut render_ctx = render::RenderContext::new(game_name, event_loop)?;

	let basecolor_only_set_layout = render_ctx.get_transparency_renderer().get_base_color_only_set_layout();
	let mut mesh_manager = component::mesh::MeshManager::new(basecolor_only_set_layout.clone());

	let light_manager = component::light::LightManager::new(&mut render_ctx)?;

	let vk_dev = render_ctx.descriptor_set_allocator().device().clone();
	let mut pbr_pipeline_config = material::pbr::PBR::get_pipeline_config(vk_dev.clone())?;
	pbr_pipeline_config.set_layouts.push(light_manager.get_all_lights_set().layout().clone());
	mesh_manager.load_set_layout("PBR", pbr_pipeline_config.set_layouts[0].clone());
	render_ctx.load_material_pipeline("PBR", pbr_pipeline_config)?;

	let (world, sky) = load_world(start_map)?;

	world.add_unique(Canvas::new(&mut render_ctx, 1280, 720)?);
	world.add_unique(render::skybox::Skybox::new(&mut render_ctx, sky)?);
	world.add_unique(CameraManager::new(&mut render_ctx, CameraFov::Y(1.0_f32.to_degrees()))?);
	world.add_unique(mesh_manager);
	world.add_unique(InputHelperWrapper { inner: WinitInputHelper::new() });
	world.add_unique(render_ctx);
	world.add_unique(light_manager);

	Ok(world)
}

// returns true if the application should exit
fn handle_event(world: &mut World, event: &mut Event<()>) -> Result<bool, GenericEngineError>
{
	world.run(|mut input_helper_wrapper: UniqueViewMut<InputHelperWrapper>| {
		input_helper_wrapper.inner.update(event);
	});

	match event {
		Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => return Ok(true),
		Event::WindowEvent { event: WindowEvent::ScaleFactorChanged { inner_size_writer, .. }, .. } => {
			// We don't want the image to be upscaled by the OS, so we tell it here that that the inner size of the
			// window in physical pixels should be exactly the same (dot-by-dot) as the swapchain's image extent.
			// It would look blurry if we don't do this.
			let extent = world.run(|render_ctx: UniqueView<RenderContext>| render_ctx.swapchain_dimensions());
			inner_size_writer.request_inner_size(extent.into())?;
		}
		Event::AboutToWait => {
			// Game logic: run systems usually specific to custom components in a project
			if world.contains_workload("Game logic") {
				world.run_workload("Game logic")?;
			}

			// Pre-render: update GPU resources for various components, to reflect the changes made in game logic systems
			world.run_workload("Pre-render")?;

			// Main rendering: build the command buffers, then submit them for presentation
			world.run_workload("Render")?;
		}
		_ => (),
	}

	Ok(false)
}

#[derive(Deserialize)]
struct WorldData
{
	pub sky: String,
	pub entities: Vec<Vec<Box<dyn component::EntityComponent>>>,
}
fn load_world(file: &str) -> Result<(World, String), GenericEngineError>
{
	let world_data: WorldData = serde_yaml::from_reader(File::open(file)?)?;
	let mut world = World::new();
	let mut systems = BTreeMap::new();
	let mut prerender_systems = BTreeMap::new();

	for entity in world_data.entities {
		let eid = world.add_entity(());
		for component in entity {
			let type_id = component.type_id();

			// add the relevant system if the component returns one
			if let Some(add_system) = component.add_system() {
				if !systems.contains_key(&type_id) {
					systems.insert(type_id, add_system);
					log::debug!("inserted system for {}", component.type_name());
				}
			}

			if let Some(add_system) = component.add_prerender_system() {
				if !prerender_systems.contains_key(&type_id) {
					prerender_systems.insert(type_id, add_system);
					log::debug!("inserted pre-render system for {}", component.type_name());
				}
			}

			component.add_to_entity(&mut world, eid);
		}
	}

	if systems.len() > 0 {
		systems
			.into_values()
			.fold(Workload::new("Game logic"), |w, s| w.with_system(s))
			.add_to_world(&world)?;
	}

	prerender_systems
		.into_values()
		.fold(Workload::new("Pre-render"), |w, s| w.with_system(s))
		.add_to_world(&world)?;

	// TODO: clean up removed components

	Workload::new("Render")
		.with_try_system(|mut render_ctx: UniqueViewMut<render::RenderContext>| render_ctx.submit_async_transfers())
		.with_try_system(draw_shadows)
		.with_try_system(draw_3d)
		.with_try_system(draw_3d_transparent_moments)
		.with_try_system(draw_3d_transparent)
		.with_try_system(draw_ui)
		.with_try_system(submit_frame)
		.add_to_world(&world)?;

	Ok((world, world_data.sky))
}

// Render shadow maps.
fn draw_shadows(
	render_ctx: UniqueView<render::RenderContext>,
	transforms: View<component::Transform>,
	mesh_manager: UniqueView<component::mesh::MeshManager>,
	mut light_manager: UniqueViewMut<component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let dir_light_extent = light_manager.get_dir_light_shadow().image().extent();
	let viewport_extent = [ dir_light_extent[0], dir_light_extent[1] ];
	let shadow_pipeline = light_manager.get_shadow_pipeline().clone();
	let shadow_format = Some(light_manager.get_dir_light_shadow().format());

	for layer_projview in light_manager.get_dir_light_projviews() {
		let mut cb = render_ctx.gather_commands(&[], shadow_format, viewport_extent)?;

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
	transforms: View<component::Transform>,
	mesh_manager: &component::mesh::MeshManager,
	pipeline: &Arc<GraphicsPipeline>,
	transparency_pass: bool,
	base_color_only: bool,
) -> Result<(), GenericEngineError>
{
	let projview = camera_manager.projview();

	for (eid, transform) in transforms.iter().with_id() {
		if (mesh_manager.has_opaque_materials(eid) && !transparency_pass) 
			|| (mesh_manager.has_transparency(eid) && transparency_pass) {
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
	render_ctx: UniqueView<render::RenderContext>,
	skybox: UniqueView<render::skybox::Skybox>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<component::Transform>,
	mesh_manager: UniqueView<component::mesh::MeshManager>,
	light_manager: UniqueView<component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let color_formats = [ Format::R16G16B16A16_SFLOAT ];
	let vp_extent = render_ctx.swapchain_dimensions();
	let light_set = vec![ light_manager.get_all_lights_set().clone() ];

	let mut cb = render_ctx.gather_commands(&color_formats, Some(render::MAIN_DEPTH_FORMAT), vp_extent)?;

	// Draw the skybox. This will effectively clear the color image.
	skybox.draw(&mut cb, camera_manager.sky_projview())?;
	
	let pbr_pipeline = render_ctx.get_pipeline("PBR").ok_or("PBR pipeline not loaded!")?;

	cb
		.bind_pipeline_graphics(pbr_pipeline.clone())?
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
	render_ctx: UniqueView<render::RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<component::Transform>,
	mesh_manager: UniqueView<component::mesh::MeshManager>,
) -> Result<(), GenericEngineError>
{
	let color_formats = [ Format::R32G32B32A32_SFLOAT, Format::R32_SFLOAT, Format::R32_SFLOAT ];
	let vp_extent = render_ctx.swapchain_dimensions();
	let mut cb = render_ctx.gather_commands(&color_formats, Some(render::MAIN_DEPTH_FORMAT), vp_extent)?;

	// This will bind the pipeline for you, since it doesn't need to do anything 
	// specific to materials (it only reads the alpha channel of each texture).
	let pipeline = render_ctx
		.get_transparency_renderer()
		.get_moments_pipeline();

	cb.bind_pipeline_graphics(pipeline.clone())?;

	draw_common(&mut cb, &camera_manager, transforms, &mesh_manager, pipeline, true, true)?;

	render_ctx
		.get_transparency_renderer()
		.add_transparency_moments_cb(cb.build()?);

	Ok(())
}

// Draw the transparent objects.
fn draw_3d_transparent(
	render_ctx: UniqueView<render::RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<component::Transform>,
	mesh_manager: UniqueView<component::mesh::MeshManager>,
	light_manager: UniqueView<component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let color_formats = [ Format::R16G16B16A16_SFLOAT, Format::R8_UNORM ];
	let vp_extent = render_ctx.swapchain_dimensions();
	let pipeline = render_ctx.get_transparency_pipeline("PBR").ok_or("PBR transparency pipeline not loaded!")?;
	let common_sets = vec![
		light_manager.get_all_lights_set().clone(), 
		render_ctx.get_transparency_renderer().get_stage3_inputs().clone()
	];

	let mut cb = render_ctx.gather_commands(&color_formats, Some(render::MAIN_DEPTH_FORMAT), vp_extent)?;

	cb
		.bind_pipeline_graphics(pipeline.clone())?
		.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 1, common_sets)?;

	draw_common(&mut cb, &camera_manager, transforms, &mesh_manager, pipeline, true, false)?;

	render_ctx
		.get_transparency_renderer()
		.add_transparency_cb(cb.build()?);

	Ok(())
}

// Draw UI elements.
fn draw_ui(
	render_ctx: UniqueView<render::RenderContext>,
	mut canvas: UniqueViewMut<ui::canvas::Canvas>,
	ui_transforms: View<ui::UITransform>,
) -> Result<(), GenericEngineError>
{
	let vp_extent = render_ctx.swapchain_dimensions();
	let mut cb = render_ctx.gather_commands(&[ Format::R16G16B16A16_SFLOAT ], None, vp_extent)?;

	cb.bind_pipeline_graphics(canvas.get_pipeline().clone())?;

	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	for (eid, _) in ui_transforms.iter().with_id() {
		// TODO: how do we respect the render order?
		canvas.draw(&mut cb, eid)?;
	}

	canvas.add_cb(cb.build()?);
	Ok(())
}

fn submit_frame(
	mut render_ctx: UniqueViewMut<render::RenderContext>,
	mut canvas: UniqueViewMut<ui::canvas::Canvas>,
	mut light_manager: UniqueViewMut<component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	render_ctx.submit_frame(canvas.take_cb(), light_manager.drain_dir_light_cb())
}

// Get data path, set up logging, and return the data path.
fn setup_log(org_name: &str, game_name: &str) -> Result<PathBuf, GenericEngineError>
{
	let data_path = dirs::data_dir()
		.ok_or("Failed to get data directory")?
		.join(org_name)
		.join(game_name);
	println!("Using data directory: {}", data_path.display());

	// Create the game data directory. Log, config, and save data files will be saved here.
	std::fs::create_dir_all(&data_path).or_else(|e| Err(format!("Failed to create data directory: {}", e)))?;

	// open log file
	let log_file_path = data_path.join("game.log");
	let log_file = std::fs::File::create(&log_file_path)
		.or_else(|e| Err(format!("Failed to create '{}': {}", log_file_path.display(), e)))?;

	// set up logger
	let logger_config = ConfigBuilder::new()
		.set_time_offset_to_local()
		.unwrap_or_else(|config_builder| {
			println!("WARNING: simplelog::ConfigBuilder::set_time_offset_to_local failed! Using UTC+0 instead.");
			config_builder
		})
		.set_time_format_rfc3339()
		.build();

	// Debug messages are disabled in release builds via the `log` crate's max level feature in Cargo.toml.
	let term_logger = TermLogger::new(
		LevelFilter::Debug,
		logger_config.clone(),
		TerminalMode::Mixed,
		ColorChoice::Auto,
	);
	let write_logger = WriteLogger::new(LevelFilter::Debug, logger_config, log_file);
	CombinedLogger::init(vec![term_logger, write_logger])?;

	log::info!("--- Initializing MithrilEngine... ---");

	Ok(data_path)
}

fn log_error(e: GenericEngineError)
{
	let mut error_string = format!("{}", e);
	log::debug!("top level error: {:?}", e);
	let mut next_err_source = e.source();
	while let Some(source) = next_err_source {
		error_string += &format!("\ncaused by: {}", source);
		log::debug!("caused by: {:?}", source);
		next_err_source = source.source();
	}
	if log::log_enabled!(log::Level::Error) {
		log::error!("{}", error_string);
	} else {
		println!("{}", error_string);
	}
	msgbox::create("Engine Error", &error_string, msgbox::common::IconType::Error)
		.unwrap_or_else(|mbe| log::error!("Failed to create error message box: {}", mbe));
}
