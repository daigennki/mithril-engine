use shipyard::{IntoIter, IntoWorkloadSystem, WorkloadSystem, UniqueView, View, ViewMut};
use serde::Deserialize;
use mithrilengine::render::RenderContext;
use mithrilengine::component::{ui, EntityComponent, WantsSystemAdded};
use mithrilengine_derive::EntityComponent;

// This is an example implementation of a framerate counter.

#[derive(shipyard::Component, Deserialize, EntityComponent)]
struct FpsCounter;
impl WantsSystemAdded for FpsCounter
{
	fn add_system(&self) -> Option<WorkloadSystem>
	{
		Some(update_fps_counter.into_workload_system().unwrap())
	}
	fn add_prerender_system(&self) -> Option<WorkloadSystem>
	{
		None
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

