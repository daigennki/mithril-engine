layout(push_constant) uniform pc
{
    mat4 projviewmodel;	// pre-multiplied projection, view, and model transformation matrices
};

layout(location = 0) in vec3 position;

void main()
{
	gl_Position = projviewmodel * vec4(position, 1.0);
}
