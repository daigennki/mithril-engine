#version 450

layout(push_constant) uniform pc
{
    mat4 projviewmodel;	// pre-multiplied projection, view, and model transformation matrices
	mat3x4 transform;	// the model transformation matrix with scale and rotation; translation stored in 4th row
};


layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;

layout(location = 0) out vec2 texcoord;
layout(location = 1) out vec3 normal_transformed;
layout(location = 2) out vec3 world_pos;
layout(location = 3) flat out int instance_index;

void main()
{
    gl_Position = projviewmodel * vec4(pos, 1.0);
	texcoord = uv;

	mat3 transform_notranslate = mat3(transform);
	vec3 translation = vec3(transform[0].w, transform[1].w, transform[2].w);

	normal_transformed = transform_notranslate * normal;
	world_pos = transform_notranslate * pos + translation;

	instance_index = gl_InstanceIndex;
}
