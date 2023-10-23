use shipyard::{IntoIter, IntoWorkloadSystem, WorkloadSystem, UniqueView, View, ViewMut};
use serde::Deserialize;
use mithrilengine::component::{EntityComponent, WantsSystemAdded};
use mithrilengine_derive::EntityComponent;

// This is an example implementation of a camera controller that would be implemented by the game developer.

#[derive(shipyard::Component, Deserialize, EntityComponent)]
struct CameraController;
impl WantsSystemAdded for CameraController
{
	fn add_system(&self) -> Option<(std::any::TypeId, WorkloadSystem)>
	{
		Some((std::any::TypeId::of::<Self>(), update_controllable_camera.into_workload_system().unwrap()))
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

		for (mut transform, _, _) in (&mut transforms, &cameras, &camera_controller).iter() {
			let sensitivity = 0.05;
			transform.rotation.z += (sensitivity * delta.0) as f32;
			while transform.rotation.z >= 360.0 || transform.rotation.z <= -360.0 {
				transform.rotation.z %= 360.0;
			}

			transform.rotation.x += (-sensitivity * delta.1) as f32;
			transform.rotation.x = transform.rotation.x.clamp(-80.0, 80.0);
		}
	}
}

