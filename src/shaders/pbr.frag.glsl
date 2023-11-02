#version 450

#ifdef TRANSPARENCY_PASS
#include "mboit_weights.glsl"
#endif

layout(binding = 0) uniform sampler sampler0;
layout(binding = 1) uniform texture2D base_color;

layout(location = 0) in vec2 texcoord;
layout(location = 1) in vec3 normal;

// If `TRANSPARENCY_PASS` is defined, the outputs in the OIT shader file included above will be used.
#ifndef TRANSPARENCY_PASS
layout(location = 0) out vec4 color_out;
#endif

/* Environment light */
/*struct DirLight
{
	vec3 direction;
	float _filler1;
	vec3 color;
	float _filler2;
};
layout(set = 1, binding = 0) uniform dir_light_ubo
{
	DirLight dir_light;
}
*/

vec3 calc_diff(vec3 light_direction, vec3 normal, vec3 tex_diffuse)
{
	float diff_intensity = max(dot(normal, light_direction), 0.0);
	vec3 diffuse = diff_intensity * tex_diffuse;
	return diffuse;
}
// calculate directional light (sunlight/moonlight)
vec3 calc_dl(vec3 tex_diffuse, vec3 normal)
{
	vec3 light_direction = normalize(vec3(1.0, 1.0, 5.0));
	vec3 light_color = vec3(12.5, 11.25, 10.0) / 8.0;
	/*vec3 light_direction = dir_light.direction;
	vec3 light_color = dir_light.color;*/
	vec3 ambient = tex_diffuse * 0.04;

	vec3 color_out = calc_diff(light_direction, normal, tex_diffuse);
	color_out += ambient;
	return light_color * color_out;
}

void main()
{
	vec4 tex_color = texture(sampler2D(base_color, sampler0), texcoord);

#ifdef TRANSPARENCY_PASS
	tex_color.rgb *= tex_color.a;
#endif

	vec3 shaded = calc_dl(tex_color.rgb, normal);

#ifdef TRANSPARENCY_PASS
	vec4 with_alpha = vec4(shaded, tex_color.a);
	write_transparent_pixel(with_alpha);
#else
	color_out = vec4(shaded, 1.0);
#endif
}

