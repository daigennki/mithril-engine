use mithril_engine::component::{ComponentSystems, EntityComponent};
use mithril_engine::{InputHelperWrapper, SystemBundle};
use mithril_engine_derive::EntityComponent;
use serde::Deserialize;
use shipyard::{IntoIter, IntoWorkloadSystem, UniqueView, View, ViewMut, WorkloadSystem};
use winit::event::MouseButton;

// This is an example implementation of a camera controller that would be implemented by the game developer.

#[derive(shipyard::Component, Deserialize, EntityComponent)]
struct CameraController;
impl ComponentSystems for CameraController
{
	fn update() -> Option<WorkloadSystem>
	{
		Some(update_controllable_camera.into_workload_system().unwrap())
	}
}
fn update_controllable_camera(
	input_helper_wrapper: UniqueView<InputHelperWrapper>,
	mut transforms: ViewMut<mithril_engine::component::Transform>,
	cameras: View<mithril_engine::component::camera::Camera>,
	camera_controller: View<CameraController>,
)
{
	let InputHelperWrapper(input_helper) = input_helper_wrapper.as_ref();
	if input_helper.mouse_held(MouseButton::Right) {
		let delta = input_helper.mouse_diff();
		let sensitivity = 0.05;
		let adjusted_delta_x = (sensitivity * delta.0) as f64;
		let adjusted_delta_y = (-sensitivity * delta.1) as f64;

		for (mut transform, _, _) in (&mut transforms, &cameras, &camera_controller).iter() {
			transform.rotation.z += adjusted_delta_x;
			transform.rotation.x += adjusted_delta_y;
			transform.rotation.x = transform.rotation.x.clamp(-80.0, 80.0);
		}
	}
}
