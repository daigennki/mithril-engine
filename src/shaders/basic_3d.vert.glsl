#version 450

layout(push_constant) uniform pc
{
    mat4 projviewmodel;	// pre-multiplied projection, view, and model transformation matrices
	mat3x4 transform;	// the model transformation matrix with scale and rotation; translation stored in 4th row
	uvec2 viewport_extent;
};


layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;

layout(location = 0) out vec2 texcoord;
layout(location = 1) out vec3 normal_transformed;
layout(location = 2) out vec3 world_pos;
layout(location = 3) flat out int instance_index;
layout(location = 4) out vec2 screen_coord_pixels;

void main()
{
    gl_Position = projviewmodel * vec4(pos, 1.0);
	texcoord = uv;

	mat3 transform_notranslate = mat3(transform);
	vec3 translation = vec3(transform[0].w, transform[1].w, transform[2].w);

	normal_transformed = transform_notranslate * normal;
	world_pos = transform_notranslate * pos + translation;

	instance_index = gl_InstanceIndex;

	if (viewport_extent.x > 0) {
		vec2 screen_coord = (gl_Position.xy / gl_Position.w) * 0.5 + 0.5;
		screen_coord_pixels = screen_coord * vec2(viewport_extent);
	} else {
		screen_coord_pixels = vec2(0.0, 0.0);
	}
}
