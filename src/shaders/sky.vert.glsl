#version 460

layout(push_constant) uniform pc
{
	mat4 sky_projview;
};

// The cube used for the skybox. It consists of 6 triangles for two opposite corners of the cube,
// drawn like two fans. (For compatibility with portability subset devices, we don't use the
// 'triangle fan' primitive topology.)
//
// Relative to camera at default state, -X is left, +Y is forward, and +Z is up.
const vec3 POSITIONS[8] = {
	{ -1.0, -1.0, -1.0 },
	{ -1.0, -1.0, 1.0 },
	{ 1.0, -1.0, 1.0 },
	{ 1.0, -1.0, -1.0 },
	{ 1.0, 1.0, -1.0 },
	{ -1.0, 1.0, -1.0 },
	{ -1.0, 1.0, 1.0 },
	{ 1.0, 1.0, 1.0 },
};
const int INDICES[36] = {
	0, 1, 2,
	0, 2, 3,
	0, 3, 4,
	0, 4, 5,
	0, 5, 6,
	0, 6, 1,
	7, 1, 2,
	7, 2, 3,
	7, 3, 4,
	7, 4, 5,
	7, 5, 6,
	7, 6, 1,
};

layout(location = 0) out vec3 cube_pos; // give original vertex position to fragment shader

void main()
{
	int index = INDICES[gl_VertexIndex];
	cube_pos = POSITIONS[index];
	vec4 new_pos = sky_projview * vec4(cube_pos, 1.0);
	gl_Position = new_pos.xyww;
} 
