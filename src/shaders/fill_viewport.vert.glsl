 #version 450

layout(push_constant) uniform pc
{
	uvec2 viewport_extent;
};

layout(location = 0) out vec2 texcoord_pixels;

void main()
{
	vec2 texcoords[3] = { { 0.0, 0.0 }, { 0.0, 2.0 }, { 2.0, 0.0 } };
	vec2 texcoord = texcoords[min(gl_VertexIndex, 2)];
	texcoord_pixels = texcoord * vec2(viewport_extent);
	gl_Position = vec4(texcoord * 2.0 - 1.0, 0.0, 1.0);
}
