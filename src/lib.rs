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
use shipyard::{UniqueView, UniqueViewMut, Workload, World};
use simplelog::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::DeviceOwned;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit_input_helper::WinitInputHelper;

use component::camera::{CameraFov, CameraManager};
use component::ui::canvas::Canvas;
use render::RenderContext;

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
			return;
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

	if let Err(e) = event_loop.run(move |mut event, window_target| match handle_event(&mut world, &mut event) {
		Ok(true) => window_target.exit(),
		Ok(false) => (),
		Err(e) => {
			log_error(e);
			window_target.exit();
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

fn init_world(
	org_name: &str,
	game_name: &str,
	start_map: &str,
	event_loop: &winit::event_loop::EventLoop<()>,
) -> Result<World, GenericEngineError>
{
	setup_log(org_name, game_name)?;

	let mut render_ctx = render::RenderContext::new(game_name, event_loop)?;

	let basecolor_only_set_layout = render_ctx.get_transparency_renderer().get_base_color_only_set_layout();
	let mut mesh_manager = component::mesh::MeshManager::new(basecolor_only_set_layout.clone());

	let light_manager = component::light::LightManager::new(&mut render_ctx)?;

	let vk_dev = render_ctx.descriptor_set_allocator().device().clone();
	let mut pbr_pipeline_config = material::pbr::PBR::get_pipeline_config(vk_dev.clone())?;
	pbr_pipeline_config
		.set_layouts
		.push(light_manager.get_all_lights_set().layout().clone());
	mesh_manager.load_set_layout("PBR", pbr_pipeline_config.set_layouts[0].clone());
	render_ctx.load_material_pipeline("PBR", pbr_pipeline_config)?;

	let (world, sky) = load_world(start_map)?;

	world.add_unique(Canvas::new(&mut render_ctx, 1280, 720)?);
	world.add_unique(render::skybox::Skybox::new(&mut render_ctx, sky)?);
	world.add_unique(CameraManager::new(&mut render_ctx, CameraFov::Y(1.0_f32.to_degrees()))?);
	world.add_unique(mesh_manager);
	world.add_unique(InputHelperWrapper {
		inner: WinitInputHelper::new(),
	});
	world.add_unique(render_ctx);
	world.add_unique(light_manager);

	Ok(world)
}

// returns true if the application should exit
fn handle_event(world: &mut World, event: &mut Event<()>) -> Result<bool, GenericEngineError>
{
	world.run(|mut wrapper: UniqueViewMut<InputHelperWrapper>| wrapper.inner.update(event));

	match event {
		Event::WindowEvent {
			event: WindowEvent::CloseRequested,
			..
		} => return Ok(true),
		Event::WindowEvent {
			event: WindowEvent::ScaleFactorChanged { inner_size_writer, .. },
			..
		} => {
			// We don't want the image to be upscaled by the OS, so we tell it here that that the inner size of the
			// window in physical pixels should be exactly the same (dot-by-dot) as the swapchain's image extent.
			// It would look blurry if we don't do this.
			let extent = world.run(|render_ctx: UniqueView<RenderContext>| render_ctx.swapchain_dimensions());
			inner_size_writer.request_inner_size(extent.into())?;
		}
		Event::WindowEvent {
			event: WindowEvent::KeyboardInput { event: key_event, .. },
			..
		} => {
			if !key_event.repeat && !key_event.state.is_pressed() {
				match key_event.physical_key {
					PhysicalKey::Code(KeyCode::F12) => {
						// Toggle fullscreen
						world.run(|r_ctx: UniqueView<RenderContext>| r_ctx.set_fullscreen(!r_ctx.is_fullscreen()));
					}
					_ => (),
				}
			}
		}
		Event::AboutToWait => {
			// Game logic: run systems usually specific to custom components in a project
			if world.contains_workload("Game logic") {
				world.run_workload("Game logic")?;
			}

			// Pre-render: update GPU resources for various components, to reflect the changes made in game logic systems
			world.run_workload("Pre-render")?;

			// Main rendering: build the command buffers, then submit them for presentation
			world.run_workload(render::workload::render)?;
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

	world.add_workload(render::workload::render);

	Ok((world, world_data.sky))
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
	std::fs::create_dir_all(&data_path).map_err(|e| format!("Failed to create data directory: {e}"))?;

	// open log file
	let log_file_path = data_path.join("game.log");
	let log_file =
		std::fs::File::create(&log_file_path).map_err(|e| format!("Failed to create '{}': {}", log_file_path.display(), e))?;

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
	let mut error_string = format!("{e}");
	log::debug!("top level error: {e:?}");

	let mut next_err_source = e.source();
	while let Some(src) = next_err_source {
		error_string += &format!("\ncaused by: {src}");
		log::debug!("caused by: {src:?}");
		next_err_source = src.source();
	}

	if log::log_enabled!(log::Level::Error) {
		log::error!("{error_string}");
	} else {
		println!("{error_string}");
	}

	if let Err(mbe) = msgbox::create("Engine Error", &error_string, msgbox::common::IconType::Error) {
		log::error!("Failed to create error message box: {mbe}");
	}
}
