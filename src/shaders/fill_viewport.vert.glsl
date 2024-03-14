#version 460
const vec2 POSITIONS[3] = { { -1.0, -1.0 }, { -1.0, 3.0 }, { 3.0, -1.0 } };
void main()
{
	gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
}
