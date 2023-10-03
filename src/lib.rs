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
	EntityId, Get, UniqueView, UniqueViewMut, View, ViewMut, Workload, WorkloadModificator, World,
};
use simplelog::*;
use std::fs::File;
use std::path::{Path, PathBuf};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer};
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};

use component::camera::{Camera, CameraFov, CameraManager};
use component::ui;
use component::ui::canvas::Canvas;
use component::DeferGpuResourceLoading;
use render::RenderContext;

#[cfg(feature = "egui")]
use egui_renderer::EguiRenderer;

type GenericEngineError = Box<dyn std::error::Error + Send + Sync>;

struct GameContext
{
	//pref_path: String,
	world: World,

	right_mouse_button_pressed: bool,
	camera_rotation: Vec3,

	#[cfg(feature = "egui")]
	egui_renderer: EguiRenderer,

	fps_ui_ent: EntityId,
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

		let mut world = load_world(&mut render_ctx, start_map)?;

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

		// add some UI entities for testing
		let dim = render_ctx.swapchain_dimensions();
		world.add_unique(Canvas::new(1280, 720, dim[0], dim[1])?);
		let fps_ui_ent = world.add_entity(ui::new_text(&mut render_ctx, "0 fps".to_string(), 32.0, [500, -320].into())?);

		// TODO: give the user a way to specify a skybox through the YAML map file
		world.add_unique(render::skybox::Skybox::new(&mut render_ctx, "sky/Daylight Box_*.png".into())?);
		world.add_unique(camera_manager);
		world.add_unique(render_ctx);

		Ok(GameContext {
			//pref_path,
			world,

			#[cfg(feature = "egui")]
			egui_renderer,

			right_mouse_button_pressed: false,
			camera_rotation: Vec3::ZERO,
			fps_ui_ent,
		})
	}

	pub fn handle_event(&mut self, event: &mut Event<()>) -> Result<(), GenericEngineError>
	{
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
			Event::DeviceEvent {
				event: DeviceEvent::Button { button: 1, state },
				..
			} => {
				self.right_mouse_button_pressed = match state {
					ElementState::Pressed => true,
					ElementState::Released => false,
				};
			}
			Event::DeviceEvent {
				event: DeviceEvent::MouseMotion { delta },
				..
			} => {
				/*if self.right_mouse_button_pressed {
					let sensitivity = 0.05;
					self.camera_rotation.z += (sensitivity * delta.0) as f32;
					while self.camera_rotation.z >= 360.0 || self.camera_rotation.z <= -360.0 {
						self.camera_rotation.z %= 360.0;
					}

					self.camera_rotation.x += (-sensitivity * delta.1) as f32;
					self.camera_rotation.x = self.camera_rotation.x.clamp(-80.0, 80.0);

					let rot_rad = self.camera_rotation * std::f32::consts::PI / 180.0;
					let rot_quat = Quat::from_euler(EulerRot::ZYX, rot_rad.z, rot_rad.y, rot_rad.x);
					let rotated = rot_quat.mul_vec3(Vec3::new(0.0, 1.0, 0.0));
					let pos = Vec3::new(0.0, 0.0, 3.0);
					let target = pos + rotated;
					let (mut render_ctx, mut camera) = self
						.world
						.borrow::<(UniqueViewMut<_>, UniqueViewMut<Camera>)>()?;
					camera.set_pos_and_target(pos, target, &mut render_ctx)?;
				}*/
			}
			Event::MainEventsCleared => {
				self.world.run_default()?; // main rendering (build the secondary command buffers)
				self.draw_debug()?;
				self.world
					.run(|mut render_ctx: UniqueViewMut<RenderContext>| render_ctx.submit_frame())?;
			}
			_ => (),
		}
		Ok(())
	}

	/// Draw some debug stuff, mostly GUI overlays.
	fn draw_debug(&mut self) -> Result<(), GenericEngineError>
	{
		self.world.run(
			|mut render_ctx: UniqueViewMut<RenderContext>,
			 mut texts: ViewMut<ui::text::Text>|
			 -> Result<(), GenericEngineError> {
				// draw the fps counter
				let delta_time = render_ctx.delta().as_secs_f64();
				let fps = 1.0 / delta_time.max(0.000001);
				let delta_ms = 1000.0 * delta_time;
				(&mut texts)
					.get(self.fps_ui_ent)
					.unwrap()
					.set_text(format!("{:.0} fps ({:.1} ms)", fps, delta_ms), &mut render_ctx)?;

				Ok(())
			},
		)?;

		#[cfg(feature = "egui")]
		self.egui_renderer.draw(&mut self.world)?;

		Ok(())
	}
}

#[derive(Deserialize)]
struct WorldData
{
	uniques: Vec<Box<dyn component::UniqueComponent>>,
	entities: Vec<Vec<Box<dyn component::EntityComponent>>>,
}
impl Into<World> for WorldData
{
	fn into(self) -> World
	{
		let mut world = World::new();
		for unique in self.uniques {
			unique.add_to_world(&mut world);
		}
		for entity in self.entities {
			let eid = world.add_entity(());
			for component in entity {
				component.add_to_entity(&mut world, eid);
			}
		}
		world
	}
}
fn load_world(render_ctx: &mut render::RenderContext, file: &str) -> Result<World, GenericEngineError>
{
	let world_data: WorldData = serde_yaml::from_reader(File::open(file)?)?;
	let world: World = world_data.into();

	// This will become the default workload, as the docs say:
	// > The default workload will automatically be set to the first workload added.
	Workload::new("Render loop")
		.with_try_system(prepare_primary_render)
		.with_try_system(draw_3d)
		.with_try_system(draw_3d_transparent_moments)
		.with_try_system(draw_3d_transparent)
		.with_try_system(draw_ui)
		.after_all(prepare_primary_render)
		.add_to_world(&world)?;

	// finish loading GPU resources for components
	// TODO: maybe figure out a way to get trait objects from shipyard
	world.run(
		|mut transforms: ViewMut<component::Transform>,
		 mut meshes: ViewMut<component::mesh::Mesh>|
		 -> Result<(), GenericEngineError> {
			for t in (&mut transforms).iter() {
				t.finish_loading(render_ctx)?;
			}
			for m in (&mut meshes).iter() {
				m.finish_loading(render_ctx)?;
			}
			Ok(())
		},
	)?;

	Ok(world)
}

fn prepare_primary_render(
	mut render_ctx: UniqueViewMut<render::RenderContext>,
	cameras: View<Camera>,
	transforms: View<component::Transform>,
	mut canvas: UniqueViewMut<Canvas>,
	mut ui_transforms: ViewMut<ui::Transform>,
	mut camera_manager: UniqueViewMut<CameraManager>,
) -> Result<(), GenericEngineError>
{
	if render_ctx.window_resized() {
		let d = render_ctx.swapchain_dimensions();
		canvas.on_screen_resize(d[0], d[1]);
		for t in (&mut ui_transforms).iter() {
			t.update_projection(render_ctx.as_mut(), canvas.projection())?;
		}
	} else {
		// Update the projection matrix on UI `Transform` components,
		// for entities that have been inserted since last time.
		for t in ui_transforms.inserted_mut().iter() {
			t.update_projection(render_ctx.as_mut(), canvas.projection())?;
		}
	}

	let active_camera_id = camera_manager.active_camera();
	if let Ok(cam) = cameras.get(active_camera_id) {
		if let Ok(t) = transforms.get(active_camera_id) {
			camera_manager.update(&mut render_ctx, t.position(), t.rotation_quat(), cam.fov)?;
		}
	}

	Ok(())
}
fn draw_3d(
	render_ctx: UniqueView<render::RenderContext>,
	skybox: UniqueView<render::skybox::Skybox>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<component::Transform>,
	meshes: View<component::mesh::Mesh>,
) -> Result<(), GenericEngineError>
{
	let mut command_buffer = render_ctx.record_main_draws()?;

	// Draw the skybox. This will effectively clear the framebuffer.
	skybox.draw(&mut command_buffer, &camera_manager)?;

	// Draw 3D objects.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	render_ctx.get_pipeline("PBR")?.bind(&mut command_buffer);
	camera_manager.push_projview(&mut command_buffer)?;
	let projview = camera_manager.projview();
	for (eid, transform) in transforms.iter().with_id() {
		if let Ok(c) = meshes.get(eid) {
			if c.has_opaque_materials() {
				transform.bind_descriptor_set(&mut command_buffer)?;

				let transform_mat = projview * transform.get_matrix();
				c.draw(&mut command_buffer, &transform_mat, false)?;
			}
		}
	}

	render_ctx.add_cb(command_buffer.build()?);
	Ok(())
}
fn draw_transparent_common(
	command_buffer: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
	camera_manager: &CameraManager,
	transforms: View<component::Transform>,
	meshes: View<component::mesh::Mesh>,
) -> Result<(), GenericEngineError>
{
	// Draw the transparent objects.
	camera_manager.push_projview(command_buffer)?;
	let projview = camera_manager.projview();
	for (eid, transform) in transforms.iter().with_id() {
		if let Ok(c) = meshes.get(eid) {
			if c.has_transparency() {
				transform.bind_descriptor_set(command_buffer)?;

				let transform_mat = projview * transform.get_matrix();
				c.draw(command_buffer, &transform_mat, true)?;
			}
		}
	}
	Ok(())
}
fn draw_3d_transparent_moments(
	render_ctx: UniqueView<render::RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<component::Transform>,
	meshes: View<component::mesh::Mesh>,
) -> Result<(), GenericEngineError>
{
	let mut command_buffer = render_ctx.record_transparency_moments_draws()?;
	draw_transparent_common(&mut command_buffer, &camera_manager, transforms, meshes)?;
	render_ctx.add_transparency_moments_cb(command_buffer.build()?);
	Ok(())
}

fn draw_3d_transparent(
	render_ctx: UniqueView<render::RenderContext>,
	camera_manager: UniqueView<CameraManager>,
	transforms: View<component::Transform>,
	meshes: View<component::mesh::Mesh>,
) -> Result<(), GenericEngineError>
{
	// Draw the transparent objects.
	let mut command_buffer = render_ctx.record_transparency_draws(render_ctx.get_pipeline("PBR")?)?;
	draw_transparent_common(&mut command_buffer, &camera_manager, transforms, meshes)?;
	render_ctx.add_transparency_cb(command_buffer.build()?);
	Ok(())
}
fn draw_ui(
	render_ctx: UniqueView<render::RenderContext>,
	ui_transforms: View<ui::Transform>,
	ui_meshes: View<ui::mesh::Mesh>,
	texts: View<ui::text::Text>,
) -> Result<(), GenericEngineError>
{
	let mut command_buffer = render_ctx.record_ui_draws()?;
	render_ctx.get_pipeline("UI")?.bind(&mut command_buffer);

	// Draw UI elements.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	for (eid, t) in ui_transforms.iter().with_id() {
		t.bind_descriptor_set(&mut command_buffer)?;

		// draw UI meshes
		// TODO: how do we respect the render order?
		if let Ok(c) = ui_meshes.get(eid) {
			c.draw(&mut command_buffer)?;
		}
		if let Ok(c) = texts.get(eid) {
			c.draw(&mut command_buffer)?;
		}
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
