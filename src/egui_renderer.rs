/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use egui_winit_vulkano::egui;
use shipyard::{EntitiesView, EntityId, Get, UniqueView, UniqueViewMut, ViewMut, World};
use std::sync::Arc;

use crate::component;
use crate::render::RenderContext;
use crate::GenericEngineError;

pub struct EguiRenderer
{
	egui_gui: egui_winit_vulkano::Gui,
	selected_ent: EntityId,
}
impl EguiRenderer
{
	pub fn new(render_ctx: &mut RenderContext, event_loop: &winit::event_loop::EventLoop<()>) -> Self
	{
		let subpass = render_ctx.get_main_render_pass().first_subpass();
		let egui_gui = egui_winit_vulkano::Gui::new_with_subpass(
			event_loop,
			render_ctx.get_surface(),
			Some(vulkano::format::Format::R16G16B16A16_SFLOAT),
			render_ctx.get_queue(),
			subpass,
		);

		EguiRenderer {
			egui_gui,
			selected_ent: Default::default(),
		}
	}

	pub fn update(&mut self, winit_event: &winit::event::WindowEvent) -> bool
	{
		self.egui_gui.update(winit_event)
	}

	pub fn draw(&mut self, world: &mut World) -> Result<(), GenericEngineError>
	{
		// set egui debug UI layout
		self.egui_gui.begin_frame();
		let egui_ctx = self.egui_gui.context();
		let mut style = egui::style::Style::default();
		style.visuals.window_shadow = egui::epaint::Shadow::default();
		style.visuals.popup_shadow = egui::epaint::Shadow::default();
		egui_ctx.set_style(Arc::new(style));

		egui::Window::new("Object list").show(&egui_ctx, |wnd| self.generate_egui_entity_list(wnd, world));
		egui::Window::new("Components")
			.show(&egui_ctx, |wnd| self.components_window_layout(wnd, world))
			.and_then(|response| response.inner)
			.transpose()?;

		// draw egui
		let render_ctx = world.borrow::<UniqueView<RenderContext>>()?;
		let egui_cb = self.egui_gui.draw_on_subpass_image(render_ctx.swapchain_dimensions());
		render_ctx.add_ui_cb(egui_cb);

		Ok(())
	}

	/// Generate the entity list window. Returns an EntityId of the newly selected entity, if one was selected.
	fn generate_egui_entity_list(&mut self, obj_window: &mut egui::Ui, world: &mut World)
	{
		world.run(|ents: EntitiesView| {
			egui::ScrollArea::vertical().show(obj_window, |obj_scroll| {
				for ent in ents.iter() {
					let label = obj_scroll.selectable_label(ent == self.selected_ent, format!("Entity {}", ent.index()));
					if label.clicked() {
						self.selected_ent = ent;
					}
				}
			});
		});
	}

	/// Set up the window layout to show the components of the currently selected entity.
	fn components_window_layout(&mut self, wnd: &mut egui::Ui, world: &mut World) -> Result<(), GenericEngineError>
	{
		let mut render_ctx = world.borrow::<UniqueViewMut<RenderContext>>()?;

		let mut meshes = world.borrow::<ViewMut<component::mesh::Mesh>>()?;
		if let Ok(mesh) = (&mut meshes).get(self.selected_ent) {
			mesh.show_egui(wnd, &mut render_ctx)?;
		}

		let mut transforms = world.borrow::<ViewMut<component::Transform>>()?;
		if let Ok(transform) = (&mut transforms).get(self.selected_ent) {
			transform.show_egui(wnd, &mut render_ctx)?;
		}

		Ok(())
	}
}
