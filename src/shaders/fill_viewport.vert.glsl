#version 460

void main()
{
	vec2 texcoords[3] = { { 0.0, 0.0 }, { 0.0, 2.0 }, { 2.0, 0.0 } };
	vec2 texcoord = texcoords[min(gl_VertexIndex, 2)];
	gl_Position = vec4(texcoord * 2.0 - 1.0, 0.0, 1.0);
}
