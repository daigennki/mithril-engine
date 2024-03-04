/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */
pub mod component;
pub mod material;
pub mod render;

use shipyard::{UniqueViewMut, Workload, WorkloadSystem, World};
use simplelog::*;
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

type Result<T> = std::result::Result<T, EngineError>;

/// Same as `run_game`, but this will get the version of your application using the
/// `CARGO_PKG_VERSION_*` environment variables. It only takes `org_name`, `app_name`, and
/// `start_map`.
#[macro_export]
macro_rules! run {
	($org_name:expr, $app_name:expr, $start_map:expr) => {
		let app_version = vulkano::Version {
			major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
			minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
			patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
		};
		mithril_engine::run_game($org_name, $app_name, $start_map, app_version)
	};
}

/// Run the game. This should be called from your `main.rs`.
///
/// - `org_name` and `app_name` will be used for the data directory.
/// - `app_name` will also be used for the window title.
/// - `start_map` is the first map to be loaded.
/// - `app_version` is the version of your application.
///
pub fn run_game(org_name: &str, app_name: &str, start_map: &str, app_version: Version)
{
	let log_config = ConfigBuilder::new()
		.set_time_offset_to_local()
		.unwrap_or_else(|config_builder| {
			println!("simplelog `set_time_offset_to_local` failed, using UTC+0 instead.");
			config_builder
		})
		.set_time_format_rfc3339()
		.build();

	// Debug messages are disabled in release builds via the `log` crate's max level feature in Cargo.toml.
	let term_logger = TermLogger::new(LevelFilter::Debug, log_config.clone(), TerminalMode::Mixed, ColorChoice::Auto);
	let loggers: Vec<Box<dyn SharedLogger>> = if std::env::args().any(|arg| arg == "--logfile") {
		match File::create("game.log") {
			Ok(log_file) => vec![term_logger, WriteLogger::new(LevelFilter::Info, log_config, log_file)],
			Err(e) => {
				eprintln!("failed to create log file: {e}");
				vec![term_logger]
			}
		}
	} else {
		vec![term_logger]
	};
	CombinedLogger::init(loggers).unwrap();

	if let Err(e) = run_game_inner(org_name, app_name, start_map, app_version) {
		log_error(e.as_ref());
	}
}
fn run_game_inner(
	org_name: &str,
	app_name: &str,
	start_map: &str,
	app_version: Version,
) -> std::result::Result<(), Box<dyn Error>>
{
	let event_loop = EventLoop::new()?;
	event_loop.set_control_flow(ControlFlow::Poll);

	let mut world = init_world(org_name, app_name, app_version, &event_loop)?;

	// Load the first map from `start_map`.
	MapData::new(Path::new(start_map))?.into_world(&mut world);

	event_loop.run(move |mut event, window_target| match handle_event(&mut world, &mut event) {
		Ok(true) => window_target.exit(),
		Ok(false) => (),
		Err(e) => {
			log_error(e.custom_error().unwrap().as_ref());
			window_target.exit();
		}
	})?;

	Ok(())
}

#[derive(shipyard::Unique, Default)]
pub struct InputHelperWrapper(pub WinitInputHelper);

/// Initialize the world with the uniques and systems that components will need.
fn init_world(_org_name: &str, app_name: &str, app_version: Version, event_loop: &EventLoop<()>) -> crate::Result<World>
{
	let mut render_ctx = render::RenderContext::new(app_name, app_version, event_loop)?;
	let viewport_extent = render_ctx.window_dimensions();

	let light_manager = render::lighting::LightManager::new(&mut render_ctx)?;
	let light_set_layout = light_manager.get_all_lights_set().layout().clone();

	let world = World::new();
	world.add_unique(render::ui::Canvas::new(&mut render_ctx, 1280, 720)?);
	world.add_unique(CameraManager::new(viewport_extent, CameraFov::Y(1.0_f64.to_degrees())));
	world.add_unique(render::model::MeshManager::new(&mut render_ctx, light_set_layout)?);
	world.add_unique(light_manager);
	world.add_unique(InputHelperWrapper::default());
	world.add_unique(render_ctx);
	world.add_unique(component::physics::PhysicsManager::default());

	// add the systems that components need
	let mut update_workload = Workload::new("update");
	let mut late_update_workload = Workload::new("late_update");
	for system_bundle in inventory::iter::<SystemBundle> {
		if let Some(system) = (system_bundle.update)() {
			log::debug!("inserting `update` system for {}", system_bundle.component_name);
			update_workload = update_workload.with_system(system);
		}
		if let Some(system) = (system_bundle.late_update)() {
			log::debug!("inserting `late_update` system for {}", system_bundle.component_name);
			late_update_workload = late_update_workload.with_system(system);
		}
	}

	update_workload.add_to_world(&world).unwrap();

	Workload::new("physics")
		.with_system(component::physics::simulate_physics)
		.add_to_world(&world)
		.unwrap();

	late_update_workload.add_to_world(&world).unwrap();

	// Main rendering: build the command buffers, then submit them for presentation.
	Workload::new("render")
		.with_try_system(render::submit_transfers)
		.with_try_system(render::model::draw_shadows)
		.with_try_system(render::model::draw_3d)
		.with_try_system(render::model::draw_3d_oit)
		.with_try_system(render::draw_ui)
		.with_try_system(render::submit_frame)
		.add_to_world(&world)
		.unwrap();

	Ok(world)
}

pub struct SystemBundle
{
	pub component_name: &'static str,
	pub update: &'static (dyn Fn() -> Option<WorkloadSystem> + Send + Sync),
	pub late_update: &'static (dyn Fn() -> Option<WorkloadSystem> + Send + Sync),
}
inventory::collect!(SystemBundle);

#[derive(serde::Deserialize)]
struct MapData
{
	sky: String,
	entities: Vec<Vec<Box<dyn component::EntityComponent>>>,
}
impl MapData
{
	/// Deserialize a map file into this struct.
	fn new(file: &Path) -> crate::Result<Self>
	{
		log::info!("Loading world map file '{}'...", file.display());
		let map_file = File::open(file).map_err(|e| EngineError::new("failed to open world map file", e))?;
		serde_yaml::from_reader(map_file).map_err(|e| EngineError::new("failed to parse world map file", e))
	}

	/// Use the data in this struct to add entities and components into the given world.
	fn into_world(self, world: &mut World)
	{
		world.run(|mut render_ctx: UniqueViewMut<render::RenderContext>| {
			if let Err(e) = render_ctx.set_skybox(self.sky.clone()) {
				log::error!("failed to set skybox to `{}`: {}", &self.sky, e);
			}
		});

		for entity in self.entities {
			let eid = world.add_entity(());
			for component in entity {
				component.add_to_entity(world, eid);
			}
		}
	}
}

// returns true if the application should exit
fn handle_event(world: &mut World, event: &mut Event<()>) -> std::result::Result<bool, shipyard::error::RunWorkload>
{
	world.run(|mut wrapper: UniqueViewMut<InputHelperWrapper>| wrapper.0.update(event));

	match event {
		Event::WindowEvent {
			event: WindowEvent::CloseRequested,
			..
		} => return Ok(true),
		Event::WindowEvent { event: window_event, .. } => {
			world.run(|mut r_ctx: UniqueViewMut<render::RenderContext>| r_ctx.handle_window_event(window_event));
		}
		Event::AboutToWait => {
			world.run_workload("update")?;
			world.run_workload("physics")?;
			world.run_workload("late_update")?;
			world.run_workload("render")?;
		}
		_ => (),
	}

	Ok(false)
}

fn log_error(e: &dyn Error)
{
	log::debug!("error debug: {e:#?}");
	log::error!("{e}");
	if let Err(mbe) = msgbox::create("Engine Error", &e.to_string(), msgbox::common::IconType::Error) {
		log::error!("failed to create error message box: {mbe}");
	}
}

#[derive(Debug)]
pub enum EngineError
{
	SourceContext
	{
		source: Box<dyn Error + Send + Sync + 'static>,
		context: &'static str,
	},
	WrapError(Box<dyn Error + Send + Sync + 'static>),
}
impl EngineError
{
	pub fn new<E>(context: &'static str, error: E) -> Self
	where
		E: Error + Send + Sync + 'static,
	{
		Self::SourceContext {
			source: Box::new(error),
			context,
		}
	}
}
impl std::fmt::Display for EngineError
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		match self {
			Self::SourceContext { source, context } => write!(f, "{context}: {source}"),
			Self::WrapError(e) => match e.source() {
				Some(source) => write!(f, "{e}: {source}"),
				None => write!(f, "{e}"),
			},
		}
	}
}
impl Error for EngineError
{
	fn source(&self) -> Option<&(dyn Error + 'static)>
	{
		match self {
			Self::SourceContext { source, .. } => Some(source.as_ref()),
			Self::WrapError(e) => e.source(),
		}
	}
}
impl From<&'static str> for EngineError
{
	fn from(string: &'static str) -> Self
	{
		Self::WrapError(string.into())
	}
}
impl From<Box<ValidationError>> for EngineError
{
	fn from(error: Box<ValidationError>) -> Self
	{
		panic!("{error}");
	}
}
impl<E: Into<Self>> From<Validated<E>> for EngineError
{
	fn from(error: Validated<E>) -> Self
	{
		error.unwrap().into()
	}
}
impl From<VulkanError> for EngineError
{
	fn from(error: VulkanError) -> Self
	{
		Self::SourceContext {
			source: Box::new(error),
			context: "a Vulkan error has occurred",
		}
	}
}
impl From<AllocateImageError> for EngineError
{
	fn from(error: AllocateImageError) -> Self
	{
		match error {
			AllocateImageError::AllocateMemory(MemoryAllocatorError::AllocateDeviceMemory(inner)) => Self::SourceContext {
				context: "allocating device memory for the image failed",
				source: Box::new(inner.unwrap()),
			},
			other => Self::WrapError(Box::new(other)),
		}
	}
}
impl From<AllocateBufferError> for EngineError
{
	fn from(error: AllocateBufferError) -> Self
	{
		match error {
			AllocateBufferError::AllocateMemory(MemoryAllocatorError::AllocateDeviceMemory(inner)) => Self::SourceContext {
				context: "allocating device memory for the buffer failed",
				source: Box::new(inner.unwrap()),
			},
			other => Self::WrapError(Box::new(other)),
		}
	}
}
