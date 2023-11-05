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
use shipyard::{iter::{IntoIter, IntoWithId}, Get, UniqueView, UniqueViewMut, View, Workload, World};
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

use component::camera::{Camera, CameraFov, CameraManager};
use component::ui;
use component::ui::canvas::Canvas;
use render::RenderContext;

//#[cfg(feature = "egui")]
//use egui_renderer::EguiRenderer;

type GenericEngineError = Box<dyn std::error::Error + Send + Sync>;

/// Run the game. This should go in your `main.rs`.
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

	match GameContext::new(org_name, game_name, start_map, &event_loop) {
		Ok(mut gctx) => {
			if let Err(e) = event_loop.run(move |mut event, window_target| {
				match event {
					Event::WindowEvent {
						event: WindowEvent::CloseRequested,
						..
					} => window_target.exit(),
					_ => (),
				};

				if let Err(e) = gctx.handle_event(&mut event) {
					log_error(e);
					window_target.exit();
				}
			}) {
				log_error(Box::new(e));
			}
		}
		Err(e) => log_error(e),
	}
}

#[derive(shipyard::Unique)]
pub struct InputHelperWrapper
{
	pub inner: WinitInputHelper,
}

struct GameContext
{
	world: World,
}
impl GameContext
{
	pub fn new(
		org_name: &str,
		game_name: &str,
		start_map: &str,
		event_loop: &winit::event_loop::EventLoop<()>,
	) -> Result<Self, GenericEngineError>
	{
		setup_log(org_name, game_name)?;

		log::info!("--- Initializing MithrilEngine... ---");

		let mut render_ctx = render::RenderContext::new(game_name, event_loop)?;
		let vk_dev = render_ctx.descriptor_set_allocator().device().clone();

		let dim = render_ctx.swapchain_dimensions();
		let canvas = Canvas::new(&mut render_ctx, 1280, 720, dim[0], dim[1])?;

		let basecolor_only_set_layout = render_ctx.get_transparency_renderer().get_base_color_only_set_layout();
		let mut mesh_manager = component::mesh::MeshManager::new(basecolor_only_set_layout.clone());

		let light_manager = component::light::LightManager::new(&mut render_ctx)?;

		let mut pbr_pipeline_config = material::pbr::PBR::get_pipeline_config(vk_dev.clone())?;
		pbr_pipeline_config.set_layouts.push(light_manager.get_all_lights_set().layout().clone());
		mesh_manager.load_set_layout("PBR", pbr_pipeline_config.set_layouts[0].clone());
		render_ctx.load_material_pipeline("PBR", pbr_pipeline_config)?;

		let (world, sky) = load_world(start_map)?;

		let mut camera_manager = CameraManager::new(&mut render_ctx, CameraFov::Y(180.0 / std::f32::consts::PI))?;
		world.run(|cameras: View<Camera>| {
			// for now, just choose the first camera in the world.
			// TODO: figure out a better way to choose the default camera
			if let Some((eid, _)) = cameras.iter().with_id().next() {
				camera_manager.set_active(eid);
			}
		});

		world.add_unique(canvas);
		world.add_unique(render::skybox::Skybox::new(&mut render_ctx, sky)?);
		world.add_unique(camera_manager);
		world.add_unique(mesh_manager);
		world.add_unique(InputHelperWrapper { inner: WinitInputHelper::new() });
		world.add_unique(render_ctx);
		world.add_unique(light_manager);

		Ok(GameContext { world })
	}

	pub fn handle_event(&mut self, event: &mut Event<()>) -> Result<(), GenericEngineError>
	{
		self.world.run(|mut input_helper_wrapper: UniqueViewMut<InputHelperWrapper>| {
			input_helper_wrapper.inner.update(event);
		});

		match event {
			Event::WindowEvent { 
				event: WindowEvent::ScaleFactorChanged { scale_factor, inner_size_writer },
				..
			} => {
				let swapchain_dimensions = self
					.world
					.run(|render_ctx: UniqueView<RenderContext>| render_ctx.swapchain_dimensions());
				let desired_physical_size =
					winit::dpi::PhysicalSize::new(swapchain_dimensions[0], swapchain_dimensions[1]);
				log::info!(
					"`ScaleFactorChanged` event gave us a scale factor of {}, giving back {:?}...",
					scale_factor,
					desired_physical_size
				);
				inner_size_writer.request_inner_size(desired_physical_size)?;
			}
			Event::WindowEvent { 
				event: WindowEvent::Resized(new_inner_size),
				..
			} => {
				log::info!("Window resized to {:?}, changing swapchain dimensions...", new_inner_size);
				self.world
					.run(|mut render_ctx: UniqueViewMut<RenderContext>| render_ctx.resize_swapchain())?;
			}
			Event::AboutToWait => {
				if self.world.contains_workload("Game logic") {
					self.world.run_workload("Game logic")?;
				}

				// main rendering: build the command buffers, then submit them for presentation
				self.world.run_workload("Render")?;
			}
			_ => (),
		}
		Ok(())
	}
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

	for entity in world_data.entities {
		let eid = world.add_entity(());
		for component in entity {
			// add the relevant system if the component returns one
			if let Some(add_system) = component.add_system() {
				let type_id = component.type_id();
				if !systems.contains_key(&type_id) {
					systems.insert(type_id, add_system);
					log::debug!("inserted system for {}", component.type_name());
				}
			}

			component.add_to_entity(&mut world, eid);
		}
	}

	if systems.len() > 0 {
		let mut workload = Workload::new("Game logic");
		for (_, system) in systems {
			workload = workload.with_system(system);
		}
		if let Err(e) = workload.add_to_world(&world) {
			log::error!("Failed to add game logic workload to world: {}", e);
		}
	}

	Workload::new("Render")
		.with_try_system(prepare_primary_render)
		.with_try_system(prepare_ui)
		.with_try_system(submit_async_transfers)
		.with_try_system(draw_3d)
		.with_try_system(draw_3d_transparent_moments)
		.with_try_system(draw_3d_transparent)
		.with_try_system(draw_ui)
		.with_try_system(submit_frame)
		.add_to_world(&world)?;

	Ok((world, world_data.sky))
}

fn prepare_primary_render(
	mut render_ctx: UniqueViewMut<render::RenderContext>,
	transforms: View<component::Transform>,
	mut camera_manager: UniqueViewMut<CameraManager>,
	cameras: View<Camera>,
	) -> Result<(), GenericEngineError>
{
	let active_camera_id = camera_manager.active_camera();
	if let Ok((t, cam)) = (&transforms, &cameras).get(active_camera_id) {
		camera_manager.update(&mut render_ctx, t.position, &t.rotation_quat(), cam.fov)?;
	}

	// TODO: clean up removed components

	Ok(())
}
fn prepare_ui(
	mut render_ctx: UniqueViewMut<render::RenderContext>,
	mut canvas: UniqueViewMut<Canvas>,
	ui_transforms: View<ui::UITransform>,
	ui_meshes: View<ui::mesh::Mesh>,
	ui_texts: View<ui::text::UIText>,
) -> Result<(), GenericEngineError>
{
	if render_ctx.window_resized() {
		let d = render_ctx.swapchain_dimensions();
		canvas.on_screen_resize(d[0], d[1]);

		for (eid, (t, mesh)) in (&ui_transforms, &ui_meshes).iter().with_id() {
			canvas.update_mesh(&mut render_ctx, eid, t, mesh)?;
		}
		for (eid, (t, text)) in (&ui_transforms, &ui_texts).iter().with_id() {
			canvas.update_text(&mut render_ctx, eid, t, text)?
		}
	} else {
		// Update inserted or modified components
		// TODO: this might run `update_mesh` or `update_text` twice when both the `Transform` and
		// the other component are inserted or modified; make it not run twice in such a case!
		for (eid, (t, mesh)) in (ui_transforms.inserted_or_modified(), &ui_meshes).iter().with_id() {
			canvas.update_mesh(&mut render_ctx, eid, t, mesh)?;
		}
		for (eid, (t, text)) in (ui_transforms.inserted_or_modified(), &ui_texts).iter().with_id() {
			canvas.update_text(&mut render_ctx, eid, t, text)?
		}
		for (eid, (t, mesh)) in (&ui_transforms, ui_meshes.inserted_or_modified()).iter().with_id() {
			canvas.update_mesh(&mut render_ctx, eid, t, mesh)?;
		}
		for (eid, (t, text)) in (&ui_transforms, ui_texts.inserted_or_modified()).iter().with_id() {
			canvas.update_text(&mut render_ctx, eid, t, text)?
		}
	}

	Ok(())
}
fn submit_async_transfers(mut render_ctx: UniqueViewMut<render::RenderContext>) -> Result<(), GenericEngineError>
{
	render_ctx.submit_async_transfers()?;
	Ok(())
}

fn draw_common(
	command_buffer: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
	camera_manager: &CameraManager,
	transforms: View<component::Transform>,
	mesh_manager: &component::mesh::MeshManager,
	//meshes: View<component::mesh::Mesh>,
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
				transparency_pass,
				base_color_only,
			)?;
		}
	}
	Ok(())
}
fn draw_3d(
	render_ctx: UniqueView<render::RenderContext>,
	skybox: UniqueView<render::skybox::Skybox>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<component::Transform>,
	mesh_manager: UniqueView<component::mesh::MeshManager>,
	light_manager: UniqueView<component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	let mut command_buffer = render_ctx.new_secondary_command_buffer(
		vec![ Some(vulkano::format::Format::R16G16B16A16_SFLOAT) ], 
		Some(render::MAIN_DEPTH_FORMAT),
		render_ctx.swapchain_dimensions()
	)?;

	// Draw the skybox. This will effectively clear the framebuffer.
	skybox.draw(&mut command_buffer, camera_manager.sky_projview())?;

	// Draw 3D objects.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	let pbr_pipeline = render_ctx.get_pipeline("PBR")?;

	// Bind the descriptor set for the lights
	command_buffer
		.bind_pipeline_graphics(pbr_pipeline.clone())?
		.push_constants(pbr_pipeline.layout().clone(), 0, camera_manager.projview())?
		.bind_descriptor_sets(
			PipelineBindPoint::Graphics,
			pbr_pipeline.layout().clone(),
			1,
			vec![light_manager.get_all_lights_set().clone()]
		)?;

	draw_common(
		&mut command_buffer,
		&camera_manager,
		transforms,
		&mesh_manager,
		pbr_pipeline,
		false,
		false,
	)?;

	render_ctx.add_cb(command_buffer.build()?);
	Ok(())
}
fn draw_3d_transparent_moments(
	render_ctx: UniqueView<render::RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<component::Transform>,
	mesh_manager: UniqueView<component::mesh::MeshManager>,
) -> Result<(), GenericEngineError>
{
	// Start recording commands for moment-based OIT. This will bind the pipeline for you, since it doesn't need to do
	// anything specific to materials (it only reads the alpha channel of each texture).
	let mut command_buffer = render_ctx.new_secondary_command_buffer(
		vec![
			Some(Format::R32G32B32A32_SFLOAT),
			Some(Format::R32_SFLOAT),
			Some(Format::R32_SFLOAT),
		],
		Some(render::MAIN_DEPTH_FORMAT),
		render_ctx.swapchain_dimensions()
	)?;

	let pipeline = render_ctx
		.get_transparency_renderer()
		.get_moments_pipeline();

	command_buffer.bind_pipeline_graphics(pipeline.clone())?;

	draw_common(&mut command_buffer, &camera_manager, transforms, &mesh_manager, pipeline, true, true)?;

	render_ctx
		.get_transparency_renderer()
		.add_transparency_moments_cb(command_buffer.build()?);
	Ok(())
}

fn draw_3d_transparent(
	render_ctx: UniqueView<render::RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<component::Transform>,
	mesh_manager: UniqueView<component::mesh::MeshManager>,
	light_manager: UniqueView<component::light::LightManager>,
) -> Result<(), GenericEngineError>
{
	// Draw the transparent objects.
	let pipeline = render_ctx.get_transparency_pipeline("PBR")?;

	let color_formats = vec![ Some(Format::R16G16B16A16_SFLOAT), Some(Format::R8_UNORM) ];

	let mut command_buffer = render_ctx
		.new_secondary_command_buffer(color_formats, Some(render::MAIN_DEPTH_FORMAT), render_ctx.swapchain_dimensions())?;

	command_buffer
		.bind_pipeline_graphics(pipeline.clone())?
		.bind_descriptor_sets(
			PipelineBindPoint::Graphics,
			pipeline.layout().clone(),
			1,
			vec![
				light_manager.get_all_lights_set().clone(), 
				render_ctx.get_transparency_renderer().get_stage3_inputs().clone()
			]
		)?;

	draw_common(&mut command_buffer, &camera_manager, transforms, &mesh_manager, pipeline, true, false)?;

	render_ctx
		.get_transparency_renderer()
		.add_transparency_cb(command_buffer.build()?);
	Ok(())
}
fn draw_ui(
	render_ctx: UniqueView<render::RenderContext>,
	mut canvas: UniqueViewMut<ui::canvas::Canvas>,
	ui_transforms: View<ui::UITransform>,
) -> Result<(), GenericEngineError>
{
	let mut command_buffer = render_ctx.new_secondary_command_buffer(
		vec![ Some(vulkano::format::Format::R16G16B16A16_SFLOAT) ], 
		None,
		render_ctx.swapchain_dimensions()
	)?;

	command_buffer.bind_pipeline_graphics(canvas.get_pipeline().clone())?;

	// Draw UI elements.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	for (eid, _) in ui_transforms.iter().with_id() {
		// TODO: how do we respect the render order?
		canvas.draw(&mut command_buffer, eid)?;
	}

	canvas.add_cb(command_buffer.build()?);
	Ok(())
}
fn submit_frame(
	mut render_ctx: UniqueViewMut<render::RenderContext>,
	mut canvas: UniqueViewMut<ui::canvas::Canvas>,
) -> Result<(), GenericEngineError>
{
	render_ctx.submit_frame(canvas.take_cb())
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
