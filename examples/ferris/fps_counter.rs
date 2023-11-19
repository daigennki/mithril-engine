use shipyard::{IntoIter, IntoWorkloadSystem, WorkloadSystem, UniqueView, ViewMut};
use serde::Deserialize;
use mithrilengine::render::RenderContext;
use mithrilengine::component::{ui, EntityComponent, WantsSystemAdded};
use mithrilengine_derive::EntityComponent;

// This is an example implementation of a framerate counter.

#[derive(shipyard::Component, Deserialize, EntityComponent)]
struct FpsCounter
{
	#[serde(skip, default = "std::time::Instant::now")]
	last_update: std::time::Instant,
	#[serde(skip)]
	frame_time_samples: Vec<std::time::Duration>,
}
impl Default for FpsCounter
{
	fn default() -> Self 
	{
		Self {
			last_update: std::time::Instant::now(),
			frame_time_samples: Vec::with_capacity(500),
		}
	}
}
impl FpsCounter
{
	// Collect the frame times, and return `Some` with the average frame time if the text should be updated.
	fn collect_frame_time(&mut self, frame_time: std::time::Duration) -> Option<std::time::Duration>
	{
		if self.frame_time_samples.capacity() < 500 {
			self.frame_time_samples.reserve(500);
		}

		self.frame_time_samples.push(frame_time);

		let dur_since_last_update = std::time::Instant::now() - self.last_update;
		(dur_since_last_update >= std::time::Duration::from_millis(250))
			.then(|| {
				let sample_count: u32 = self.frame_time_samples.len().try_into().unwrap();
				let frame_time_avg = self.frame_time_samples.drain(..).sum::<std::time::Duration>() / sample_count;

				self.last_update = std::time::Instant::now();

				frame_time_avg
			})
	}
}
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
	mut fps_counter: ViewMut<FpsCounter>,
)
{
	for (mut text_component, counter) in (&mut texts, &mut fps_counter).iter() {
		// update the fps counter's text
		if let Some(avg_frame_time) = counter.collect_frame_time(render_ctx.delta()) {
			let delta_time = avg_frame_time.as_secs_f64();
			let fps = 1.0 / delta_time.max(0.000001);
			let delta_ms = 1000.0 * delta_time;
			text_component.text_str = format!("{:.0} fps ({:.1} ms)", fps, delta_ms);
		}
	}
}

