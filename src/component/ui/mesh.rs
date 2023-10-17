/* -----------------------------------------------------------------------------
	Copyright (c) daigennki and MithrilEngine developers.

	Licensed under the BSD 3-clause license.
	https://opensource.org/license/BSD-3-clause/
----------------------------------------------------------------------------- */

use std::path::PathBuf;

#[derive(Clone, Copy)]
pub enum MeshType 
{
	Quad,
	Frame(u32),	// `u32` is border width in logical pixels
}
impl Default for MeshType
{
	fn default() -> Self
	{
		Self::Quad
	}
}


/// UI component that renders to a mesh, such as a quad, or a background frame mesh.
#[derive(Default, shipyard::Component)]
#[track(All)]
pub struct Mesh
{
	pub mesh_type: MeshType,

	// Leave this empty if the image is set by another component, such as `Text`.
	// This has no effect (image is overridden) if another component like `Text` sets the image.
	pub image_path: PathBuf,
}
/*impl Mesh
{
	pub fn new(render_ctx: &mut RenderContext, tex: Arc<Texture>) -> Result<Self, GenericEngineError>
	{
		// resize position vertices according to texture dimensions
		let dimensions = UVec2::from_array(tex.dimensions()).as_vec2();
		let half_dimensions = dimensions * 0.5;
		Self::new_from_corners(render_ctx, -half_dimensions, half_dimensions, tex)
	}
}*/

