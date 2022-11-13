/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
pub mod component;
mod egui_renderer;
mod material;
mod render;

use glam::*;
use serde::Deserialize;
use shipyard::{
	iter::{IntoIter, IntoWithId},
	EntityId, Get, UniqueView, UniqueViewMut, View, ViewMut, Workload, WorkloadModificator, World,
};
use simplelog::*;
use std::fs::File;
use std::path::{Path, PathBuf};
use vulkano::command_buffer::SecondaryAutoCommandBuffer;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};

use component::camera::Camera;
use component::ui;
use component::ui::canvas::Canvas;
use component::DeferGpuResourceLoading;
use egui_renderer::EguiRenderer;
use render::RenderContext;

type GenericEngineError = Box<dyn std::error::Error + Send + Sync>;

struct GameContext
{
	//pref_path: String,
	world: World,

	right_mouse_button_pressed: bool,
	camera_rotation: Vec3,

	egui_renderer: EguiRenderer,

	fps_ui_ent: EntityId,
}
impl GameContext
{
	// game context "constructor"
	pub fn new(
		org_name: &str, game_name: &str, start_map: &str, event_loop: &winit::event_loop::EventLoop<()>,
	) -> Result<Self, GenericEngineError>
	{
		/*let pref_path =*/
		setup_log(org_name, game_name)?;

		log::info!("--- Initializing MithrilEngine... ---");
		// get command line arguments
		// let args: Vec<String> = std::env::args().collect();

		let mut render_ctx = render::RenderContext::new(game_name, event_loop)?;
		render_ctx.load_material_pipeline("UI.yaml")?;
		render_ctx.load_material_pipeline("PBR.yaml")?;

		let mut world = load_world(&mut render_ctx, start_map)?;

		let egui_renderer = EguiRenderer::new(&mut render_ctx, event_loop);

		// add some UI entities for testing
		let dim = render_ctx.swapchain_dimensions();
		world.add_unique(Canvas::new(1280, 720, dim[0], dim[1])?);
		let fps_ui_ent = world.add_entity(ui::new_text(&mut render_ctx, "0 fps".to_string(), 32.0, [-500, -320].into())?);

		world.add_unique(render::skybox::Skybox::new(&mut render_ctx, "sky/Daylight Box_*.png".into())?);
		world.add_unique(render_ctx);
		world.add_unique(ThreadedRenderingManager::new(8));

		Ok(GameContext {
			//pref_path,
			world,
			egui_renderer,
			right_mouse_button_pressed: false,
			camera_rotation: Vec3::ZERO,
			fps_ui_ent,
		})
	}

	pub fn handle_event(&mut self, event: &Event<()>) -> Result<(), GenericEngineError>
	{
		match event {
			Event::WindowEvent { event, .. } => {
				self.egui_renderer.update(event);
			}
			Event::DeviceEvent {
				event: DeviceEvent::Button { button: 1, state }, ..
			} => {
				self.right_mouse_button_pressed = match state {
					ElementState::Pressed => true,
					ElementState::Released => false,
				};
			}
			Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
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
				self.submit_frame()?;
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

		self.egui_renderer.draw(&mut self.world)?;

		Ok(())
	}

	fn submit_frame(&mut self) -> Result<(), GenericEngineError>
	{
		let (mut render_ctx, mut trm) = self
			.world
			.borrow::<(UniqueViewMut<RenderContext>, UniqueViewMut<ThreadedRenderingManager>)>()?;
		render_ctx.submit_frame(trm.take_built_command_buffers(), trm.take_transparency_cb().unwrap())?;
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
	let world_data: WorldData = serde_yaml::from_reader(File::open(Path::new("maps").join(file))?)?;
	let world: World = world_data.into();

	// This will become the default workload, as the docs say:
	// > The default workload will automatically be set to the first workload added.
	Workload::new("Render loop")
		.with_try_system(prepare_primary_render)
		.with_try_system(draw_3d)
		.with_try_system(draw_ui)
		.after_all(prepare_primary_render)
		.add_to_world(&world)?;

	// finish loading GPU resources for components
	// TODO: maybe figure out a way to get trait objects from shipyard
	world.run(|mut camera: UniqueViewMut<Camera>| camera.finish_loading(render_ctx))?;
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

#[derive(shipyard::Unique)]
struct ThreadedRenderingManager
{
	built_command_buffers: Vec<SecondaryAutoCommandBuffer>,
	transparency_cb: Option<SecondaryAutoCommandBuffer>,
	default_capacity: usize,
}
impl ThreadedRenderingManager
{
	pub fn new(default_capacity: usize) -> Self
	{
		ThreadedRenderingManager {
			built_command_buffers: Vec::with_capacity(default_capacity),
			transparency_cb: None,
			default_capacity,
		}
	}

	/// Add a secondary command buffer that has been built.
	pub fn add_cb(&mut self, command_buffer: SecondaryAutoCommandBuffer)
	{
		self.built_command_buffers.push(command_buffer);
	}

	pub fn add_transparency_cb(&mut self, command_buffer: SecondaryAutoCommandBuffer)
	{
		self.transparency_cb = Some(command_buffer)
	}

	/// Take all of the secondary command buffers that have been built.
	pub fn take_built_command_buffers(&mut self) -> Vec<SecondaryAutoCommandBuffer>
	{
		std::mem::replace(&mut self.built_command_buffers, Vec::with_capacity(self.default_capacity))
	}

	pub fn take_transparency_cb(&mut self) -> Option<SecondaryAutoCommandBuffer>
	{
		self.transparency_cb.take()
	}
}

fn draw_3d(
	render_ctx: UniqueView<render::RenderContext>, mut trm: UniqueViewMut<ThreadedRenderingManager>,
	skybox: UniqueView<render::skybox::Skybox>, camera: UniqueView<Camera>, transforms: View<component::Transform>,
	meshes: View<component::mesh::Mesh>,
) -> Result<(), GenericEngineError>
{
	let cur_fb = render_ctx.get_main_framebuffer();
	let mut command_buffer = render_ctx.new_secondary_command_buffer(cur_fb)?;

	// Draw the skybox. This will effectively clear the framebuffer.
	skybox.draw(&mut command_buffer, &camera)?;

	// Draw 3D objects.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	render_ctx.get_pipeline("PBR")?.bind(&mut command_buffer);

	camera.bind(&mut command_buffer)?;
	let projview = camera.get_projview();
	for (eid, transform) in transforms.iter().with_id() {
		// draw 3D meshes
		if let Ok(c) = meshes.get(eid) {
			if !c.has_transparency() {
				transform.bind_descriptor_set(&mut command_buffer)?;

				let transform_mat = projview * transform.get_matrix();
				c.draw(&mut command_buffer, &transform_mat)?;
			}
		}
	}

	trm.add_cb(command_buffer.build()?);

	draw_3d_transparent(render_ctx, trm, camera, transforms, meshes)?;

	Ok(())
}
fn draw_3d_transparent(
	render_ctx: UniqueView<render::RenderContext>, mut trm: UniqueViewMut<ThreadedRenderingManager>,
	camera: UniqueView<Camera>, transforms: View<component::Transform>, meshes: View<component::mesh::Mesh>,
) -> Result<(), GenericEngineError>
{
	let cur_fb = render_ctx.get_transparency_framebuffer();
	let mut command_buffer = render_ctx.new_secondary_command_buffer(cur_fb)?;

	// Draw the transparent objects.
	render_ctx
		.get_pipeline("PBR")?
		.bind_transparency(&mut command_buffer)?;

	camera.bind(&mut command_buffer)?;
	let projview = camera.get_projview();
	for (eid, transform) in transforms.iter().with_id() {
		// draw 3D meshes
		if let Ok(c) = meshes.get(eid) {
			if c.has_transparency() {
				transform.bind_descriptor_set(&mut command_buffer)?;

				let transform_mat = projview * transform.get_matrix();
				c.draw(&mut command_buffer, &transform_mat)?;
			}
		}
	}

	trm.add_transparency_cb(command_buffer.build()?);
	Ok(())
}
fn draw_ui(
	render_ctx: UniqueView<render::RenderContext>, mut trm: UniqueViewMut<ThreadedRenderingManager>,
	ui_transforms: View<ui::Transform>, ui_meshes: View<ui::mesh::Mesh>, texts: View<ui::text::Text>,
) -> Result<(), GenericEngineError>
{
	// Draw UI elements.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	let cur_fb = render_ctx.get_main_framebuffer();
	let mut command_buffer = render_ctx.new_secondary_command_buffer(cur_fb)?;

	render_ctx.get_pipeline("UI")?.bind(&mut command_buffer);
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

	trm.add_cb(command_buffer.build()?);
	Ok(())
}
fn prepare_primary_render(
	mut render_ctx: UniqueViewMut<render::RenderContext>, mut camera: UniqueViewMut<Camera>, mut canvas: UniqueViewMut<Canvas>,
	mut ui_transforms: ViewMut<ui::Transform>,
) -> Result<(), GenericEngineError>
{
	if render_ctx.window_resized() {
		let d = render_ctx.swapchain_dimensions();

		// If the screen size changed, update anything relying on the screen size.
		camera.update_window_size(d[0], d[1], &mut render_ctx)?;

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
			event_loop.run(move |event, _, control_flow| {
				match event {
					Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
						*control_flow = winit::event_loop::ControlFlow::Exit; // TODO: show exit confirmation dialog here
					}
					_ => (),
				};

				if let Err(e) = gctx.handle_event(&event) {
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
	let term_logger = TermLogger::new(LevelFilter::Debug, logger_config.clone(), TerminalMode::Mixed, ColorChoice::Auto);
	let write_logger = WriteLogger::new(LevelFilter::Debug, logger_config, log_file);
	CombinedLogger::init(vec![term_logger, write_logger])?;

	Ok(data_path)
}

fn log_error(e: GenericEngineError)
{
	let mut error_string = format!("{}", e);
	let mut next_err_source = e.source();
	while let Some(source) = next_err_source {
		error_string += &format!("\ncaused by: {}", source);
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
