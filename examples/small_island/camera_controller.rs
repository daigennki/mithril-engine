use mithrilengine::component::{EntityComponent, ComponentSystems};
use mithrilengine::SystemBundle;
use mithrilengine_derive::EntityComponent;
use serde::Deserialize;
use shipyard::{IntoIter, IntoWorkloadSystem, UniqueView, View, ViewMut, WorkloadSystem};

// This is an example implementation of a camera controller that would be implemented by the game developer.

#[derive(shipyard::Component, Deserialize, EntityComponent)]
struct CameraController;
impl ComponentSystems for CameraController
{
	fn update() -> Option<WorkloadSystem>
	{
		Some(update_controllable_camera.into_workload_system().unwrap())
	}
	fn late_update() -> Option<WorkloadSystem>
	{
		None
	}
}
fn update_controllable_camera(
	input_helper_wrapper: UniqueView<mithrilengine::InputHelperWrapper>,
	mut transforms: ViewMut<mithrilengine::component::Transform>,
	cameras: View<mithrilengine::component::camera::Camera>,
	camera_controller: View<CameraController>,
)
{
	let input_helper = &input_helper_wrapper.inner;
	if input_helper.mouse_held(1) {
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
