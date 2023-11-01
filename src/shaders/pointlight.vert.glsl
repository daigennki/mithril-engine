layout(push_constant) uniform pc
{
    mat4 projview;	// pre-multiplied projection and view matrices
	mat4 model_transform;	// just the model transformation matrix
};
struct PointLight 
{
	vec4 position;
	vec4 direction;
    vec3 color;
    float range;
};
layout(binding = 0) uniform cur_light
{
	PointLight light;
};

layout(location = 0) in vec3 position;

void main()
{
	vec4 output = model_transform * vec4(position, 1.0);
	float dist = length(output.xyz - light.position.xyz);	// get distance between fragment and light source
    gl_Position = projview * output;
	gl_Position.z = output.w * dist / (light.range * 2);	// map to [0;1] range by dividing by far plane
}
