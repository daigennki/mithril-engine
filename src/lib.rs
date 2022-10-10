/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
mod render;
pub mod component;
mod material;

use std::fs::File;
use std::path::{ Path, PathBuf };
use winit::event::{ Event, WindowEvent };
use simplelog::*;
use serde::Deserialize;
use shipyard::{ World, View, ViewMut, Get, EntitiesView, UniqueView, UniqueViewMut, Workload, WorkloadModificator, EntityId };
use shipyard::iter::{ IntoIter, IntoWithId };

use vulkano::command_buffer::{ RenderPassBeginInfo, SecondaryAutoCommandBuffer, SubpassContents };

use egui_winit_vulkano::egui;

use component::ui;
use component::ui::{ canvas::Canvas };
use component::camera::Camera;
use component::{ DeferGpuResourceLoading, Draw };

type GenericEngineError = Box<dyn std::error::Error + Send + Sync>;

struct GameContext
{
	//pref_path: String,
	world: World,

	egui_gui: egui_winit_vulkano::Gui,
	selected_ent: EntityId
}
impl GameContext
{
	// game context "constructor"
	pub fn new(org_name: &str, game_name: &str, start_map: &str, event_loop: &winit::event_loop::EventLoop<()>) 
		-> Result<Self, GenericEngineError>
	{
		/*let pref_path =*/ setup_log(org_name, game_name)?;

		log::info!("--- Initializing MithrilEngine... ---");
		// get command line arguments
		// let args: Vec<String> = std::env::args().collect();

		let mut render_ctx = render::RenderContext::new(game_name, &event_loop)?;
		render_ctx.load_material_pipeline("UI.yaml")?;
		render_ctx.load_material_pipeline("PBR.yaml")?;

		let mut world = load_world(&mut render_ctx, start_map)?;

		// set up egui
		let subpass = vulkano::render_pass::Subpass::from(
			render_ctx.get_current_framebuffer().render_pass().clone(), 1
		).unwrap();
		let gui = egui_winit_vulkano::Gui::new_with_subpass(render_ctx.get_surface(), None, render_ctx.get_queue(), subpass);

		// add some UI entities for testing
		let dim = render_ctx.swapchain_dimensions();
		world.add_unique(Canvas::new(1280, 720, dim[0], dim[1])?);
		//world.add_entity(ui::new_image(&mut render_ctx, "test_image.png", [ 0, 0 ].into())?);
		//world.add_entity(ui::new_text(&mut render_ctx, "Hello World!", 32.0, [ -200, -200 ].into())?);

		world.add_unique(render::skybox::Skybox::new(&mut render_ctx, "sky/Daylight Box_*.png".into())?);
		world.add_unique(render_ctx);
		world.add_unique(ThreadedRenderingManager::new(2));

		Ok(GameContext { 
			//pref_path: pref_path,
			world: world,
			egui_gui: gui,
			selected_ent: Default::default()
		})
	}

	pub fn handle_event(&mut self, event: &Event<()>) -> Result<(), GenericEngineError>
	{
		match event {
			Event::WindowEvent{ event: we, .. } => { self.egui_gui.update(we); },
			Event::MainEventsCleared => {
				// main rendering (build the secondary command buffers)
				self.world.run_default()?;

				// set debug UI layout
				self.world.run(|mut render_ctx: UniqueViewMut<render::RenderContext>| {
					render_ctx.wait_for_fence()
				})?;
				
				let mut mat_result = None;
				let mut tr_result = None;
				self.egui_gui.immediate_ui(|gui| {
					let ctx = gui.context();
					let outermost_frame = egui::containers::Frame::none()
						.inner_margin(egui::style::Margin::same(4.0))
						.fill(egui::Color32::TRANSPARENT);

					egui::CentralPanel::default()
						.frame(outermost_frame)
						.show(&ctx, |ui| {
							ui.columns(5, |columns| {
								let ui = &mut columns[0];
							});
							
							// the object list window
							egui::Window::new("Object list")
								.show(&ctx, |obj_window| {
									if let Some(s) = generate_egui_entity_list(&self.world, obj_window, self.selected_ent) {
										self.selected_ent = s;
									}
								});
							
							// the material properties window
							egui::Window::new("Material properties")
								.show(&ctx, |mat_window| {
									mat_result = Some(material_properties_window_layout(&self.world, mat_window,
									self.selected_ent));
								});

							// transform properties window
							egui::Window::new("Transform properties")
								.show(&ctx, |mat_window| {
									tr_result = Some(transform_properties_window_layout(&self.world, mat_window,
									self.selected_ent));
								});

						});
				});
				if let Some(mr) = mat_result {
					mr?;
				}
				if let Some(tr) = tr_result {
					tr?;
				}
				
				// finalize the rendering for this frame by executing the secondary command buffers
				self.world.run(| 
					mut render_ctx: UniqueViewMut<render::RenderContext>, 
					mut trm: UniqueViewMut<ThreadedRenderingManager> 
				| -> Result<(), GenericEngineError>
				{
					let egui_cb = self.egui_gui.draw_on_subpass_image(render_ctx.swapchain_dimensions());

					let mut primary_cb = render_ctx.new_primary_command_buffer()?;

					// execute the copies from staging buffers to the actual images and buffers
					if let Some(staging_cb) = render_ctx.take_staging_command_buffer()? {
						primary_cb.execute_secondary(staging_cb)?;
					}
					
					let mut rp_begin_info = RenderPassBeginInfo::framebuffer(render_ctx.get_current_framebuffer());
					rp_begin_info.clear_values = vec![ None, None ];
					primary_cb.begin_render_pass(rp_begin_info, SubpassContents::SecondaryCommandBuffers)?;
					primary_cb.execute_secondaries(trm.take_built_command_buffers())?;
					primary_cb.next_subpass(SubpassContents::SecondaryCommandBuffers)?;
					primary_cb.execute_secondary(egui_cb)?;
					primary_cb.end_render_pass()?;

					render_ctx.submit_commands(primary_cb.build()?)?;

					Ok(())
				})?;
			},
			_ => ()
		}
		Ok(())
	}
}

/// Generate the entity list for the debug UI. Returns an EntityId of the newly selected entity, if one was selected.
fn generate_egui_entity_list(world: &shipyard::World, obj_window: &mut egui::Ui, selected: EntityId) -> Option<EntityId>
{
	let mut newly_selected = None;
	world.run(|mut ents: EntitiesView| {
		egui::ScrollArea::vertical()
			.show(obj_window, |obj_scroll| {
				for ent in ents.iter() {
					if obj_scroll.selectable_label(ent == selected, format!("Entity {}", ent.index())).clicked() {
						newly_selected = Some(ent);
					}
				}
			});
	});
	newly_selected
}

fn material_properties_window_layout(
	world: &shipyard::World, mat_wnd: &mut egui::Ui, selected_ent: EntityId,
) -> Result<(), GenericEngineError>
{
	world.run(|mut render_ctx: UniqueViewMut<render::RenderContext>, 
		transforms: View<component::Transform>, 
		mut meshes: ViewMut<component::mesh::Mesh>|
	{
		for (eid, mesh) in (&mut meshes).iter().with_id() {
			if eid == selected_ent {
				if let Some(mut materials) = mesh.get_materials() {
					let mut mat = &mut materials[0];
					let mut color = mat.get_base_color().to_array();
					mat_wnd.color_edit_button_rgba_unmultiplied(&mut color);
					mat.set_base_color(color.into(), &mut render_ctx)?;
				}
				break;
			}
		}
		Ok(())
	})
}

fn transform_properties_window_layout(
	world: &shipyard::World, wnd: &mut egui::Ui, selected_ent: EntityId,
) -> Result<(), GenericEngineError>
{
	world.run(|mut render_ctx: UniqueViewMut<render::RenderContext>, 
		mut transforms: ViewMut<component::Transform>|
	{
		for (eid, transform) in (&mut transforms).iter().with_id() {
			if eid == selected_ent {
				if !transform.is_this_static() {
					let mut pos = transform.position();
					wnd.label("X");
					wnd.add(egui::DragValue::new(&mut pos.x).speed(0.1));
					wnd.label("Y");
					wnd.add(egui::DragValue::new(&mut pos.y).speed(0.1));
					wnd.label("Z");
					wnd.add(egui::DragValue::new(&mut pos.z).speed(0.1));
					transform.set_pos(pos)?;
				}

				break;
			}
		}
		Ok(())
	})
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
	world.run(|mut components: ViewMut<component::Transform>| -> Result<(), GenericEngineError> {
		for component in (&mut components).iter() {
			component.finish_loading(render_ctx)?;
		}
		Ok(())
	})?;
	world.run(|mut components: ViewMut<component::mesh::Mesh>| -> Result<(), GenericEngineError> {
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
	built_command_buffers: Vec<SecondaryAutoCommandBuffer>,
	default_capacity: usize
}
impl ThreadedRenderingManager
{
	pub fn new(default_capacity: usize) -> Self
	{
		ThreadedRenderingManager{
			built_command_buffers: Vec::with_capacity(default_capacity),
			default_capacity: default_capacity
		}
	}

	/// Add a secondary command buffer that has been built.
	pub fn add_cb(&mut self, command_buffer: SecondaryAutoCommandBuffer)
	{
		self.built_command_buffers.push(command_buffer);
	}

	/// Take all of the secondary command buffers that have been built.
	pub fn take_built_command_buffers(&mut self) -> Vec<SecondaryAutoCommandBuffer>
	{
		std::mem::replace(&mut self.built_command_buffers, Vec::with_capacity(self.default_capacity))
	}
}

fn draw_3d(
	mut render_ctx: UniqueViewMut<render::RenderContext>,
	mut trm: UniqueViewMut<ThreadedRenderingManager>,
	skybox: UniqueView<render::skybox::Skybox>,

	camera: UniqueViewMut<Camera>,
	transforms: View<component::Transform>,
	meshes: View<component::mesh::Mesh>,
) -> Result<(), GenericEngineError>
{
	let cur_fb = render_ctx.get_current_framebuffer();
	let mut command_buffer = render_ctx.new_secondary_command_buffer(cur_fb)?;
	command_buffer.set_viewport(0, [ render_ctx.get_swapchain_viewport() ]);
		
	// Draw the skybox. This will effectively clear the framebuffer.
	skybox.draw(&mut command_buffer, &camera)?;
	
	// Draw 3D objects.
	// This will ignore anything without a `Transform` component, since it would be impossible to draw without one.
	command_buffer.bind_pipeline(render_ctx.get_pipeline("PBR")?);
	
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
) -> Result<(), GenericEngineError>
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
	command_buffer.set_viewport(0, [ render_ctx.get_swapchain_viewport() ]);

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
fn prepare_primary_render(
	mut render_ctx: UniqueViewMut<render::RenderContext>,
	mut camera: UniqueViewMut<Camera>
)
	-> Result<(), GenericEngineError>
{
	let (_, new_image_dimensions) = render_ctx.next_swapchain_image()?;

	if let Some(d) = new_image_dimensions {
		camera.update_window_size(d[0], d[1])?;
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
fn setup_log(org_name: &str, game_name: &str) -> Result<PathBuf, GenericEngineError>
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
	
	// Debug messages are disabled in release builds via the `log` crate's max level feature in Cargo.toml.
	let term_logger = TermLogger::new(LevelFilter::Debug, logger_config.clone(), TerminalMode::Mixed, ColorChoice::Auto);
	let write_logger = WriteLogger::new(LevelFilter::Debug, logger_config, log_file);
	CombinedLogger::init(vec![ term_logger, write_logger ])?;

	Ok(data_path)
}

fn log_error(e: GenericEngineError)
{
	if log::log_enabled!(log::Level::Error) {
		log::error!("{}", e);
	} else {
		println!("{}", e);
	}
	msgbox::create("Engine Error", &e.to_string(), msgbox::common::IconType::Error) 
		.unwrap_or_else(|mbe| log::error!("Failed to create error message box: {}", mbe));
}

