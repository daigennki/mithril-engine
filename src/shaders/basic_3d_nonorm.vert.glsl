#version 450

layout(push_constant) uniform pc
{
    mat4 projviewmodel;	// pre-multiplied projection, view, and model transformation matrices
	mat3 transform_notranslate;	// the model transformation matrix with just scale and rotation
};

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;

layout(location = 0) out vec2 texcoord;

void main()
{
    gl_Position = projviewmodel * vec4(pos, 1.0);
	texcoord = uv;
} 
