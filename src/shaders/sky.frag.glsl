#version 460

layout(binding = 0) uniform samplerCube sky_tex;

layout(location = 0) in vec3 cube_pos;
layout(location = 0) out vec4 color_out;

void main()
{
	color_out = texture(sky_tex, cube_pos.xzy);
} 
