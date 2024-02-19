/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

pub mod component;
pub mod material;
pub mod render;

use glam::*;
use serde::Deserialize;
use shipyard::{UniqueViewMut, Workload, World};
use simplelog::*;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use vulkano::buffer::AllocateBufferError;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::image::AllocateImageError;
use vulkano::memory::allocator::MemoryAllocatorError;
use vulkano::{Validated, ValidationError, Version, VulkanError};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit_input_helper::WinitInputHelper;

use component::camera::{CameraFov, CameraManager};
use render::RenderContext;

type Result<T> = std::result::Result<T, EngineError>;

#[macro_export]
macro_rules! run {
	($org_name:expr, $game_name:expr, $start_map:expr) => {
		let app_version = vulkano::Version {
			major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
			minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
			patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
		};
		mithrilengine::run_game($org_name, $game_name, $start_map, app_version)
	};
}

/// Run the game. This should be called from your `main.rs`.
///
/// - `org_name` and `game_name` will be used for the data directory.
/// - `game_name` will also be used for the window title.
/// - `start_map` is the first map (level/world) to be loaded.
/// - `app_version` is the version of the game/application.
///
/// There is also the macro simply named `run` which will get the version of your application using
/// the `CARGO_PKG_VERSION_*` environment variables. That macro only takes `org_name`, `game_name`,
/// and `start_map`, in that order.
pub fn run_game(org_name: &str, game_name: &str, start_map: &str, app_version: Version)
{
	let event_loop = match EventLoop::new() {
		Ok(el) => el,
		Err(e) => {
			log_error(&e);
			return;
		}
	};

	event_loop.set_control_flow(ControlFlow::Poll);

	let mut world = match init_world(org_name, game_name, app_version, start_map, &event_loop) {
		Ok(w) => w,
		Err(e) => {
			log_error(&e);
			return;
		}
	};

	if let Err(e) = event_loop.run(move |mut event, window_target| match handle_event(&mut world, &mut event) {
		Ok(true) => window_target.exit(),
		Ok(false) => (),
		Err(e) => {
			log_error(&e);
			window_target.exit();
		}
	}) {
		log_error(&e);
	}
}

#[derive(shipyard::Unique, Default)]
pub struct InputHelperWrapper
{
	pub inner: WinitInputHelper,
}

fn init_world(
	org_name: &str,
	game_name: &str,
	app_version: Version,
	start_map: &str,
	event_loop: &EventLoop<()>,
) -> crate::Result<World>
{
	// Create the game data directory. Log, config, and save data files will be saved here.
	let data_path = dirs::data_dir()
		.ok_or("Failed to get data directory")?
		.join(org_name)
		.join(game_name);
	println!("Using data directory: {}", data_path.display());
	std::fs::create_dir_all(&data_path).map_err(|e| EngineError::new("Failed to create data directory", e))?;

	setup_log(&data_path);

	let mut render_ctx = render::RenderContext::new(game_name, app_version, event_loop)?;
	let viewport_extent = render_ctx.window_dimensions();

	let light_manager = render::lighting::LightManager::new(&mut render_ctx)?;
	let light_set_layout = light_manager.get_all_lights_set().layout().clone();

	let (world, sky) = load_world(start_map)?;
	render_ctx.set_skybox(sky)?;

	world.add_unique(render::ui::Canvas::new(&mut render_ctx, 1280, 720)?);
	world.add_unique(CameraManager::new(viewport_extent, CameraFov::Y(1.0_f64.to_degrees())));
	world.add_unique(render::model::MeshManager::new(&mut render_ctx, light_set_layout)?);
	world.add_unique(light_manager);
	world.add_unique(InputHelperWrapper::default());
	world.add_unique(render_ctx);

	world.add_unique(component::physics::PhysicsManager::default());

	Ok(world)
}

// returns true if the application should exit
fn handle_event(world: &mut World, event: &mut Event<()>) -> crate::Result<bool>
{
	world.run(|mut wrapper: UniqueViewMut<InputHelperWrapper>| wrapper.inner.update(event));

	match event {
		Event::WindowEvent {
			event: WindowEvent::CloseRequested,
			..
		} => return Ok(true),
		Event::WindowEvent { event: window_event, .. } => {
			world.run(|mut r_ctx: UniqueViewMut<RenderContext>| r_ctx.handle_window_event(window_event));
		}
		Event::AboutToWait => {
			// Game logic: run systems usually specific to custom components in a project.
			if world.contains_workload("Game logic") {
				world.run_workload("Game logic").unwrap();
			}

			// Pre-render: update GPU resources for various components, to reflect the changes made
			// in game logic systems.
			world.run_workload("Pre-render").unwrap();

			// Main rendering: build the command buffers, then submit them for presentation.
			world
				.run_workload(render::render_workload)
				.map_err(|e| EngineError::new("failed to run render workload", e))?;
		}
		_ => (),
	}

	Ok(false)
}

#[derive(Deserialize)]
struct WorldData
{
	sky: String,
	entities: Vec<Vec<Box<dyn component::EntityComponent>>>,
}
fn load_world(file: &str) -> crate::Result<(World, String)>
{
	log::info!("Loading world map file '{file}'...");
	let world_file = File::open(file).map_err(|e| EngineError::new("failed to open world map file", e))?;
	let world_data: WorldData =
		serde_yaml::from_reader(world_file).map_err(|e| EngineError::new("failed to parse world map file", e))?;
	let mut world = World::new();
	let mut systems = BTreeMap::new();
	let mut prerender_systems = BTreeMap::new();

	for entity in world_data.entities {
		let eid = world.add_entity(());
		for component in entity {
			let type_id = component.type_id();

			// add the relevant system if the component returns one
			if let Some(add_system) = component.add_system() {
				systems.entry(type_id).or_insert_with(|| {
					log::debug!("inserting system for {}", component.type_name());
					add_system
				});
			}

			if let Some(add_system) = component.add_prerender_system() {
				prerender_systems.entry(type_id).or_insert_with(|| {
					log::debug!("inserting pre-render system for {}", component.type_name());
					add_system
				});
			}

			component.add_to_entity(&mut world, eid);
		}
	}

	if !systems.is_empty() {
		systems
			.into_values()
			.fold(Workload::new("Game logic"), |w, s| w.with_system(s))
			.add_to_world(&world)
			.expect("failed to add game logic workload to world");
	}

	prerender_systems
		.into_values()
		.fold(Workload::new("Pre-render"), |w, s| w.with_system(s))
		.add_to_world(&world)
		.expect("failed to add pre-render workload to world");

	world.add_workload(render::render_workload);

	Ok((world, world_data.sky))
}

// Use the "force_enable_log_file" feature to always write a log file regardless of the
// "--logfile" command line argument being set or not.
#[cfg(feature = "force_enable_log_file")]
const FORCE_ENABLE_LOG_FILE: bool = true;
#[cfg(not(feature = "force_enable_log_file"))]
const FORCE_ENABLE_LOG_FILE: bool = false;

fn setup_log(data_path: &Path)
{
	let config = ConfigBuilder::new()
		.set_time_offset_to_local()
		.unwrap_or_else(|config_builder| {
			println!("WARNING: simplelog::ConfigBuilder::set_time_offset_to_local failed! Using UTC+0 instead.");
			config_builder
		})
		.set_time_format_rfc3339()
		.build();

	// Debug messages are disabled in release builds via the `log` crate's max level feature in Cargo.toml.
	let term_logger = TermLogger::new(LevelFilter::Debug, config.clone(), TerminalMode::Mixed, ColorChoice::Auto);

	let log_file_requested = std::env::args().any(|arg| arg == "--logfile");
	let loggers: Vec<Box<dyn SharedLogger>> = if log_file_requested || FORCE_ENABLE_LOG_FILE {
		let log_file_path = data_path.join("game.log");
		match std::fs::File::create(&log_file_path) {
			Ok(log_file) => vec![term_logger, WriteLogger::new(LevelFilter::Debug, config, log_file)],
			Err(e) => {
				println!("ERROR: Failed to create log file '{}': {}", log_file_path.display(), e);
				vec![term_logger]
			}
		}
	} else {
		vec![term_logger]
	};

	CombinedLogger::init(loggers).unwrap();

	log::info!("--- Initializing MithrilEngine... ---");
}

fn log_error(e: &dyn std::error::Error)
{
	log::debug!("top level error: {e:?}");
	let mut next_err_source = e.source();
	while let Some(src) = next_err_source {
		log::debug!("caused by: {src:?}");
		next_err_source = src.source();
	}

	if log::log_enabled!(log::Level::Error) {
		log::error!("{e}");
	} else {
		println!("{e}");
	}
	if let Err(mbe) = msgbox::create("Engine Error", &format!("{e}"), msgbox::common::IconType::Error) {
		log::error!("Failed to create error message box: {mbe}");
	}
}

#[derive(Debug)]
pub struct EngineError
{
	source: Option<Box<dyn Error + Send + Sync + 'static>>,
	context: &'static str,
}
impl EngineError
{
	pub fn new<E>(context: &'static str, error: E) -> Self
	where
		E: Error + Send + Sync + 'static,
	{
		Self {
			source: Some(Box::new(error)),
			context,
		}
	}
}
impl std::fmt::Display for EngineError
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		match &self.source {
			Some(e) => write!(f, "{}: {}", self.context, e),
			None => write!(f, "{}", self.context),
		}
	}
}
impl Error for EngineError
{
	fn source(&self) -> Option<&(dyn Error + 'static)>
	{
		self.source
			.as_ref()
			.map(|src_box| -> &(dyn Error + 'static) { src_box.as_ref() })
	}
}
impl From<&'static str> for EngineError
{
	fn from(string: &'static str) -> Self
	{
		Self {
			source: None,
			context: string,
		}
	}
}
impl From<Box<ValidationError>> for EngineError
{
	fn from(error: Box<ValidationError>) -> Self
	{
		panic!("{error}");
	}
}
impl From<Validated<VulkanError>> for EngineError
{
	fn from(error: Validated<VulkanError>) -> Self
	{
		Self {
			source: Some(Box::new(error.unwrap())),
			context: "a Vulkan error has occurred",
		}
	}
}
impl From<Validated<AllocateImageError>> for EngineError
{
	fn from(error: Validated<AllocateImageError>) -> Self
	{
		match error.unwrap() {
			AllocateImageError::CreateImage(source) => Self {
				context: "failed to create a Vulkan image",
				source: Some(Box::new(source)),
			},
			AllocateImageError::AllocateMemory(source) => Self {
				context: "failed to allocate memory for a Vulkan image",
				..source.into()
			},
			AllocateImageError::BindMemory(source) => Self {
				context: "failed to bind memory to a Vulkan image",
				source: Some(Box::new(source)),
			},
		}
	}
}
impl From<Validated<AllocateBufferError>> for EngineError
{
	fn from(error: Validated<AllocateBufferError>) -> Self
	{
		match error.unwrap() {
			AllocateBufferError::CreateBuffer(source) => Self {
				context: "failed to create a Vulkan buffer",
				source: Some(Box::new(source)),
			},
			AllocateBufferError::AllocateMemory(source) => Self {
				context: "failed to allocate memory for a Vulkan buffer",
				..source.into()
			},
			AllocateBufferError::BindMemory(source) => Self {
				context: "failed to bind memory to a Vulkan buffer",
				source: Some(Box::new(source)),
			},
		}
	}
}
impl From<MemoryAllocatorError> for EngineError
{
	fn from(error: MemoryAllocatorError) -> Self
	{
		let source: Box<dyn Error + Send + Sync + 'static> = match error {
			MemoryAllocatorError::AllocateDeviceMemory(inner) => Box::new(inner.unwrap()),
			other => Box::new(other),
		};

		Self {
			source: Some(source),
			context: "Vulkan memory allocation failed",
		}
	}
}
