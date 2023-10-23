/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod component;
mod material;
mod render;

#[cfg(feature = "egui")]
mod egui_renderer;

use glam::*;
use serde::Deserialize;
use shipyard::{
	iter::{IntoIter, IntoWithId},
	Get, IntoWorkloadSystem, UniqueView, UniqueViewMut, View, ViewMut, Workload, WorkloadSystem, World,
};
use simplelog::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use winit::event::{Event, WindowEvent};
use winit_input_helper::WinitInputHelper;

use component::{EntityComponent, WantsSystemAdded};
use component::camera::{Camera, CameraFov, CameraManager};
use component::ui;
use component::ui::canvas::Canvas;
use render::RenderContext;
use mithrilengine_derive::EntityComponent;

#[cfg(feature = "egui")]
use egui_renderer::EguiRenderer;

type GenericEngineError = Box<dyn std::error::Error + Send + Sync>;

#[derive(shipyard::Component, Deserialize, EntityComponent)]
struct FpsCounter;
impl WantsSystemAdded for FpsCounter
{
	fn add_system(&self) -> Option<(std::any::TypeId, WorkloadSystem)>
	{
		Some((std::any::TypeId::of::<Self>(), update_fps_counter.into_workload_system().unwrap()))
	}
}
fn update_fps_counter(
	render_ctx: UniqueView<RenderContext>,
	mut texts: ViewMut<ui::text::UIText>,
	fps_counter: View<FpsCounter>,
)
{
	for (mut text_component, _) in (&mut texts, &fps_counter).iter() {
		// update the fps counter's text
		let delta_time = render_ctx.delta().as_secs_f64();
		let fps = 1.0 / delta_time.max(0.000001);
		let delta_ms = 1000.0 * delta_time;
		text_component.text_str = format!("{:.0} fps ({:.1} ms)", fps, delta_ms);
	}
}

#[derive(shipyard::Unique)]
pub struct InputHelperWrapper
{
	pub inner: WinitInputHelper,
}

struct GameContext
{
	//pref_path: String,
	world: World,

	#[cfg(feature = "egui")]
	egui_renderer: EguiRenderer,
}
impl GameContext
{
	// game context "constructor"
	pub fn new(
		org_name: &str,
		game_name: &str,
		start_map: &str,
		event_loop: &winit::event_loop::EventLoop<()>,
	) -> Result<Self, GenericEngineError>
	{
		/*let pref_path =*/
		setup_log(org_name, game_name)?;

		log::info!("--- Initializing MithrilEngine... ---");

		let mut render_ctx = render::RenderContext::new(game_name, event_loop)?;
		render_ctx.load_material_pipeline("UI.yaml")?;
		render_ctx.load_material_pipeline("PBR.yaml")?;

		let world = load_world(start_map)?;

		#[cfg(feature = "egui")]
		let egui_renderer = EguiRenderer::new(&mut render_ctx, event_loop);

		let mut camera_manager = CameraManager::new(&mut render_ctx, CameraFov::Y(180.0 / std::f32::consts::PI))?;
		world.run(|cameras: View<Camera>| {
			// for now, just choose the first camera in the world.
			// TODO: figure out a better way to choose the default camera
			if let Some((eid, _)) = cameras.iter().with_id().next() {
				camera_manager.set_active(eid);
			}
		});

		// TODO: give the user a way to specify a skybox through the YAML map file
		let dim = render_ctx.swapchain_dimensions();
		world.add_unique(Canvas::new(&mut render_ctx, 1280, 720, dim[0], dim[1])?);
		world.add_unique(render::skybox::Skybox::new(&mut render_ctx, "sky/Daylight Box_*.png".into())?);
		world.add_unique(camera_manager);
		world.add_unique(render_ctx);
		world.add_unique(component::TransformManager::default());
		world.add_unique(component::mesh::MeshManager::default());
		world.add_unique(InputHelperWrapper { inner: WinitInputHelper::new() });

		Ok(GameContext {
			//pref_path,
			world,

			#[cfg(feature = "egui")]
			egui_renderer,
		})
	}

	pub fn handle_event(&mut self, event: &mut Event<()>) -> Result<(), GenericEngineError>
	{
		self.world.run(|mut input_helper_wrapper: UniqueViewMut<InputHelperWrapper>| {
			input_helper_wrapper.inner.update(event);
		});

		match event {
			Event::WindowEvent { event, .. } => {
				match event {
					WindowEvent::ScaleFactorChanged {
						scale_factor,
						new_inner_size,
					} => {
						let swapchain_dimensions = self
							.world
							.run(|render_ctx: UniqueView<RenderContext>| render_ctx.swapchain_dimensions());
						let desired_physical_size =
							winit::dpi::PhysicalSize::new(swapchain_dimensions[0], swapchain_dimensions[1]);
						log::info!(
							"`ScaleFactorChanged` event gave us an inner size of {:?} (scale factor {}), giving back {:?}...",
							&new_inner_size,
							scale_factor,
							desired_physical_size
						);
						**new_inner_size = desired_physical_size;
					}
					WindowEvent::Resized(new_inner_size) => {
						log::info!("Window resized to {:?}, changing swapchain dimensions...", new_inner_size);
						self.world
							.run(|mut render_ctx: UniqueViewMut<RenderContext>| render_ctx.resize_swapchain())?;
					}
					_ => (),
				}

				#[cfg(feature = "egui")]
				self.egui_renderer.update(event);
			}
			Event::MainEventsCleared => {
				if self.world.contains_workload("Game logic") {
					self.world.run_workload("Game logic")?;
				}

				self.world.run_workload("Render")?; // main rendering (build the secondary command buffers)
				//self.draw_debug()?;
				self.world.run(|mut render_ctx: UniqueViewMut<RenderContext>| render_ctx.submit_frame())?;
			}
			_ => (),
		}
		Ok(())
	}

	/*/// Draw some debug stuff, mostly GUI overlays.
	fn draw_debug(&mut self) -> Result<(), GenericEngineError>
	{
		#[cfg(feature = "egui")]
		self.egui_renderer.draw(&mut self.world)?;

		Ok(())
	}*/
}

#[derive(Deserialize)]
struct WorldData
{
	entities: Vec<Vec<Box<dyn component::EntityComponent>>>,
}
impl WorldData
{
	fn into(self) -> World
	{
		let mut world = World::new();
		let mut systems = BTreeMap::new();

		for entity in self.entities {
			let eid = world.add_entity(());
			for component in entity {
				// add the relevant system if the component returns one
				if let Some((type_id, add_system)) = component.add_system() {
					if !systems.contains_key(&type_id) {
						systems.insert(type_id, add_system);
						log::debug!("inserted system for {:?}", type_id);
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
		
		world
	}
}
fn load_world(file: &str) -> Result<World, GenericEngineError>
{
	let world_data: WorldData = serde_yaml::from_reader(File::open(file)?)?;
	let world: World = world_data.into();

	Workload::new("Render")
		.with_try_system(prepare_primary_render)
		.with_try_system(prepare_ui)
		.with_try_system(draw_3d)
		.with_try_system(draw_3d_transparent_moments)
		.with_try_system(draw_3d_transparent)
		.with_try_system(draw_ui)
		.add_to_world(&world)?;

	Ok(world)
}

fn prepare_primary_render(
	mut render_ctx: UniqueViewMut<render::RenderContext>,
	mut transform_manager: UniqueViewMut<component::TransformManager>,
	transforms: View<component::Transform>,
	mut camera_manager: UniqueViewMut<CameraManager>,
	cameras: View<Camera>,
	mut mesh_manager: UniqueViewMut<component::mesh::MeshManager>,
	meshes: View<component::mesh::Mesh>,
) -> Result<(), GenericEngineError>
{
	for (eid, t) in transforms.inserted_or_modified().iter().with_id() {
		transform_manager.update(&mut render_ctx, eid, t)?;
	}

	for (eid, mesh) in meshes.inserted().iter().with_id() {
		mesh_manager.load(&mut render_ctx, eid, mesh)?;
	}

	// TODO: clean up removed components

	let active_camera_id = camera_manager.active_camera();
	if let Ok((t, cam)) = (&transforms, &cameras).get(active_camera_id) {
		camera_manager.update(&mut render_ctx, t.position, &t.rotation_quat(), cam.fov)?;
	}

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

fn draw_common(
	command_buffer: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
	camera_manager: &CameraManager,
	transform_manager: &component::TransformManager,
	transforms: View<component::Transform>,
	mesh_manager: &component::mesh::MeshManager,
	//meshes: View<component::mesh::Mesh>,
	pipeline: &render::pipeline::Pipeline,
	transparency_pass: bool,
) -> Result<(), GenericEngineError>
{
	let projview = camera_manager.projview();
	for (eid, transform) in transforms.iter().with_id() {
		if (mesh_manager.has_opaque_materials(eid) && !transparency_pass) 
			|| (mesh_manager.has_transparency(eid) && transparency_pass) {
			command_buffer.bind_descriptor_sets(
				vulkano::pipeline::PipelineBindPoint::Graphics,
				pipeline.layout(),
				0,
				transform_manager.get_descriptor_set(eid).ok_or("transform not loaded")?.clone()
			)?;

			let transform_mat = projview * transform.get_matrix();
			mesh_manager.draw(eid, command_buffer, pipeline.layout(), &transform_mat, transparency_pass)?;
		}
	}
	Ok(())
}
fn draw_3d(
	render_ctx: UniqueView<render::RenderContext>,
	skybox: UniqueView<render::skybox::Skybox>,
	camera_manager: UniqueView<CameraManager>,
	transform_manager: UniqueView<component::TransformManager>,
	transforms: View<component::Transform>,
	mesh_manager: UniqueView<component::mesh::MeshManager>,
) -> Result<(), GenericEngineError>
{
	let mut command_buffer = render_ctx.record_main_draws()?;

	// Draw the skybox. This will effectively clear the framebuffer.
	skybox.draw(&mut command_buffer, camera_manager.sky_projview())?;

	// Draw 3D objects.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	let pbr_pipeline = render_ctx.get_pipeline("PBR")?;
	pbr_pipeline.bind(&mut command_buffer)?;

	command_buffer.push_constants(pbr_pipeline.layout(), 0, camera_manager.projview())?;

	draw_common(
		&mut command_buffer, 
		&camera_manager, 
		&transform_manager, 
		transforms, 
		&mesh_manager, 
		pbr_pipeline, 
		false
	)?;

	render_ctx.add_cb(command_buffer.build()?);
	Ok(())
}
fn draw_3d_transparent_moments(
	render_ctx: UniqueView<render::RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	transform_manager: UniqueView<component::TransformManager>,
	transforms: View<component::Transform>,
	mesh_manager: UniqueView<component::mesh::MeshManager>,
) -> Result<(), GenericEngineError>
{
	let (mut command_buffer, pipeline) = render_ctx.record_transparency_moments_draws(camera_manager.projview())?;

	draw_common(&mut command_buffer, &camera_manager, &transform_manager, transforms, &mesh_manager, pipeline, true)?;

	render_ctx.add_transparency_moments_cb(command_buffer.build()?);
	Ok(())
}

fn draw_3d_transparent(
	render_ctx: UniqueView<render::RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	transform_manager: UniqueView<component::TransformManager>,
	transforms: View<component::Transform>,
	mesh_manager: UniqueView<component::mesh::MeshManager>,
) -> Result<(), GenericEngineError>
{
	// Draw the transparent objects.
	let pipeline = render_ctx.get_pipeline("PBR")?;
	let mut command_buffer = render_ctx.record_transparency_draws(pipeline, camera_manager.projview())?;

	draw_common(&mut command_buffer, &camera_manager, &transform_manager, transforms, &mesh_manager, pipeline, true)?;

	render_ctx.add_transparency_cb(command_buffer.build()?);
	Ok(())
}
fn draw_ui(
	render_ctx: UniqueView<render::RenderContext>,
	canvas: UniqueView<ui::canvas::Canvas>,
	ui_transforms: View<ui::UITransform>,
	//ui_meshes: View<ui::mesh::Mesh>,
	//texts: View<ui::text::Text>,
) -> Result<(), GenericEngineError>
{
	let mut command_buffer = render_ctx.record_ui_draws()?;
	let pipeline = render_ctx.get_pipeline("UI")?;
	pipeline.bind(&mut command_buffer)?;

	// Draw UI elements.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	for (eid, _) in ui_transforms.iter().with_id() {
		// TODO: how do we respect the render order?
		canvas.draw(&mut command_buffer, pipeline.layout(), eid)?;
	}

	render_ctx.add_ui_cb(command_buffer.build()?);
	Ok(())
}

/// Run the game. This should go in your `main.rs`.
/// `org_name` and `game_name` will be used for the data directory.
/// `game_name` will also be used for the window title.
/// `start_map` is the first map (level/world) to be loaded.
pub fn run_game(org_name: &str, game_name: &str, start_map: &str)
{
	let event_loop = winit::event_loop::EventLoop::new();

	GameContext::new(org_name, game_name, start_map, &event_loop)
		.and_then(|mut gctx| {
			event_loop.run(move |mut event, _, control_flow| {
				match event {
					Event::WindowEvent {
						event: WindowEvent::CloseRequested,
						..
					} => {
						*control_flow = winit::event_loop::ControlFlow::Exit; // TODO: show exit confirmation dialog here
					}
					_ => (),
				};

				if let Err(e) = gctx.handle_event(&mut event) {
					log_error(e);
					*control_flow = winit::event_loop::ControlFlow::Exit;
				}
			})
		})
		.unwrap_or_else(|e| log_error(e));
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
