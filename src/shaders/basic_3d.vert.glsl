#version 450

layout(push_constant) uniform pc
{
    mat4 projviewmodel;	// pre-multiplied projection, view, and model transformation matrices
	mat3 transform_notranslate;	// the model transformation matrix with just scale and rotation
	vec3 translation;
};

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;

layout(location = 0) out vec2 texcoord;
layout(location = 1) out vec3 normal_transformed;
layout(location = 2) out vec3 world_pos;

void main()
{
    gl_Position = projviewmodel * vec4(pos, 1.0);
	texcoord = uv;
	normal_transformed = transform_notranslate * normal;
	world_pos = transform_notranslate * pos + translation;
}
