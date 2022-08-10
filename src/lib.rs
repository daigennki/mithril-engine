/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod render;
pub mod component;

use std::path::{ Path, PathBuf };
use winit::event::{ Event, WindowEvent };
use simplelog::*;
use serde::Deserialize;
use shipyard::{ World, View, ViewMut, Get, UniqueViewMut, Workload, WorkloadModificator };
use shipyard::iter::{ IntoIter, IntoWithId };

use vulkano::command_buffer::SecondaryAutoCommandBuffer;

use component::ui;
use component::ui::{ canvas::Canvas };
use component::camera::Camera;
use component::{ DeferGpuResourceLoading, Draw };

#[cfg(debug_assertions)]
use LevelFilter::Debug as EngineLogLevel;
#[cfg(not(debug_assertions))]
use LevelFilter::Info as EngineLogLevel;

struct GameContext
{
	//pref_path: String,
	world: World
}
impl GameContext
{
	// game context "constructor"
	pub fn new(org_name: &str, game_name: &str, start_map: &str, event_loop: &winit::event_loop::EventLoop<()>) 
		-> Result<Self, Box<dyn std::error::Error + Send + Sync>>
	{
		/*let pref_path =*/ setup_log(org_name, game_name)?;

		log::info!("--- Initializing MithrilEngine... ---");
		// get command line arguments
		// let args: Vec<String> = std::env::args().collect();

		let mut render_ctx = render::RenderContext::new(game_name, &event_loop)?;

		let mut world = load_world(&mut render_ctx, start_map)?;

		// add some UI entities for testing
		let dim = render_ctx.swapchain_dimensions();
		world.add_unique(Canvas::new(1280, 720, dim[0], dim[1])?);
		world.add_entity(ui::new_image(&mut render_ctx, "test_image.png", [ 0, 0 ].into())?);
		world.add_entity(ui::new_text(&mut render_ctx, "Hello World!", 32.0, [ -200, -200 ].into())?);

		world.add_unique(render_ctx);
		world.add_unique(ThreadedRenderingManager::new());

		Workload::new("Render loop")
			.with_try_system(prepare_primary_render)
			.with_try_system(draw_3d)
			.with_try_system(draw_ui)
			.with_try_system(submit_primary_render)
			.after_all(prepare_primary_render)
			.before_all(submit_primary_render)
			.add_to_world(&world)?;
		//world.set_default_workload("Render loop")?;

		Ok(GameContext { 
			//pref_path: pref_path,
			world: world
		})
	}

	pub fn handle_event(&mut self, event: &Event<()>) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
	{
		match event {
			Event::RedrawEventsCleared => self.draw_in_event_loop(),
			_ => Ok(())
		}
	}

	fn draw_in_event_loop(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
	{
		self.world.run_workload("Render loop")?;
		Ok(())
	}
}

#[derive(Deserialize)]
struct WorldData
{
	uniques: Vec<Box<dyn component::UniqueComponent>>,
	entities: Vec<Vec<Box<dyn component::EntityComponent>>>
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
fn load_world(render_ctx: &mut render::RenderContext, file: &str) -> Result<World, Box<dyn std::error::Error + Send + Sync>>
{
	let yaml_string = String::from_utf8(std::fs::read(Path::new("maps").join(file))?)?;
	let world_data: WorldData = serde_yaml::from_str(&yaml_string)?;
	let world: World = world_data.into();

	// finish loading GPU resources for components
	// TODO: maybe figure out a way to get trait objects from shipyard
	world.run(|mut component: UniqueViewMut<Camera>| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
		component.finish_loading(render_ctx)?;
		Ok(())
	})?;
	world.run(|mut components: ViewMut<component::Transform>| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
		for component in (&mut components).iter() {
			component.finish_loading(render_ctx)?;
		}
		Ok(())
	})?;
	world.run(|mut components: ViewMut<component::mesh::Mesh>| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
		for component in (&mut components).iter() {
			component.finish_loading(render_ctx)?;
		}
		Ok(())
	})?;

	Ok(world)
}

#[derive(shipyard::Unique)]
struct ThreadedRenderingManager
{
	built_command_buffers: Vec<SecondaryAutoCommandBuffer>
}
impl ThreadedRenderingManager
{
	pub fn new() -> Self
	{
		ThreadedRenderingManager{ built_command_buffers: Vec::new() }
	}

	/// Add a secondary command buffer that has been built.
	pub fn add_cb(&mut self, command_buffer: SecondaryAutoCommandBuffer)
	{
		self.built_command_buffers.push(command_buffer);
	}

	/// Take all of the secondary command buffers that have been built.
	pub fn take_built_command_buffers(&mut self) -> Vec<SecondaryAutoCommandBuffer>
	{
		std::mem::take(&mut self.built_command_buffers)
	}
}

fn draw_3d(
	mut render_ctx: UniqueViewMut<render::RenderContext>,
	mut trm: UniqueViewMut<ThreadedRenderingManager>,

	camera: UniqueViewMut<Camera>,
	transforms: View<component::Transform>,
	meshes: View<component::mesh::Mesh>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
{
	// Draw 3D objects.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	let cur_fb = render_ctx.get_current_framebuffer();
	let mut command_buffer = render_ctx.new_secondary_command_buffer(cur_fb)?;
	command_buffer.bind_pipeline(render_ctx.get_pipeline("World")?);
	camera.bind(&mut command_buffer)?;
	for (eid, transform) in transforms.iter().with_id() {
		transform.bind_descriptor_set(&mut command_buffer)?;

		// draw 3D meshes
		if let Ok(c) = meshes.get(eid) {
			c.draw(&mut command_buffer)?;
		}
	}

	trm.add_cb(command_buffer.build()?);
	Ok(())
}
fn draw_ui(
	mut render_ctx: UniqueViewMut<render::RenderContext>,
	mut trm: UniqueViewMut<ThreadedRenderingManager>,

	canvas: UniqueViewMut<Canvas>,
	mut ui_transforms: ViewMut<ui::Transform>, 
	ui_meshes: View<ui::mesh::Mesh>,
	texts: View<ui::text::Text>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
{
	// Update the projection matrix on UI `Transform` components, 
	// for entities that have been inserted since last time.
	for t in ui_transforms.inserted_mut().iter() {
		t.update_projection(render_ctx.as_mut(), canvas.projection())?;
	}

	// Draw UI elements.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	let cur_fb = render_ctx.get_current_framebuffer();
	let mut command_buffer = render_ctx.new_secondary_command_buffer(cur_fb)?;
	command_buffer.bind_pipeline(render_ctx.get_pipeline("UI")?);
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
fn prepare_primary_render(mut render_ctx: UniqueViewMut<render::RenderContext>) 
	-> Result<(), Box<dyn std::error::Error + Send + Sync>>
{
	render_ctx.next_swapchain_image()?;
	Ok(())
}
fn submit_primary_render(
	mut render_ctx: UniqueViewMut<render::RenderContext>, 
	mut trm: UniqueViewMut<ThreadedRenderingManager>
)
	-> Result<(), Box<dyn std::error::Error + Send + Sync>>
{	
	let cur_fb = render_ctx.get_current_framebuffer();
	let mut primary_cb = render_ctx.new_primary_command_buffer()?;
	let mut rp_begin_info = vulkano::command_buffer::RenderPassBeginInfo::framebuffer(cur_fb);
	rp_begin_info.clear_values = vec![
		Some(vulkano::format::ClearValue::Float([0.5, 0.9, 1.0, 1.0])),
		Some(vulkano::format::ClearValue::Depth(1.0))
	];

	primary_cb.begin_render_pass(rp_begin_info, vulkano::command_buffer::SubpassContents::SecondaryCommandBuffers)?;
	primary_cb.execute_secondaries(trm.take_built_command_buffers())?;
	primary_cb.end_render_pass()?;

	render_ctx.submit_commands(primary_cb.build()?)?;

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
		.and_then(|mut gctx| event_loop.run(move |event, _, control_flow| {
			match event {
				Event::WindowEvent{ event: WindowEvent::CloseRequested, .. } => {
					*control_flow = winit::event_loop::ControlFlow::Exit;	// TODO: show exit confirmation dialog here
				},
				_ => (),
			};
			
			if let Err(e) = gctx.handle_event(&event) {
				log_error(e);
				*control_flow = winit::event_loop::ControlFlow::Exit;
			}
		}))
		.unwrap_or_else(|e| log_error(e));
}

// Get data path, set up logging, and return the data path.
fn setup_log(org_name: &str, game_name: &str) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>>
{
	let data_path = dirs::data_dir().ok_or("Failed to get data directory")?.join(org_name).join(game_name);
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

	let term_logger = TermLogger::new(EngineLogLevel, logger_config.clone(), TerminalMode::Mixed, ColorChoice::Auto);
	let write_logger = WriteLogger::new(EngineLogLevel, logger_config, log_file);
    CombinedLogger::init(vec![ term_logger, write_logger ])?;

	Ok(data_path)
}

fn log_error(e: Box<dyn std::error::Error + Send + Sync>)
{
	if log::log_enabled!(log::Level::Error) {
		log::error!("{}", e);
	} else {
		println!("{}", e);
	}
	msgbox::create("Engine Error", &e.to_string(), msgbox::common::IconType::Error) 
		.unwrap_or_else(|mbe| log::error!("Failed to create error message box: {}", mbe));
}

