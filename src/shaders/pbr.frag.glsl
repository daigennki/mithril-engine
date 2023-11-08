#version 450

#ifdef TRANSPARENCY_PASS
#include "mboit_weights.glsl"
#endif

layout(binding = 0) uniform sampler sampler0;
layout(binding = 1) uniform texture2D base_color;

/* lighting stuff */
layout(set = 1, binding = 0) uniform samplerShadow shadow_sampler;
struct DirLight
{
	mat4 projviews[3];
	vec3 direction;
	float _filler1;
	vec4 color_intensity;
};
layout(set = 1, binding = 1) uniform dir_light_ubo
{
	DirLight dir_light;
};
layout(set = 1, binding = 2) uniform texture2D dir_light_shadow;

layout(location = 0) in vec2 texcoord;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 world_pos;

// If `TRANSPARENCY_PASS` is defined, the outputs in the OIT shader file included above will be used.
#ifndef TRANSPARENCY_PASS
layout(location = 0) out vec4 color_out;
#endif

vec3 calc_diff(vec3 light_direction, vec3 normal, vec3 tex_diffuse)
{
	vec4 dir_light_relative_pos = dir_light.projviews[0] * vec4(world_pos, 1.0);
	dir_light_relative_pos /= dir_light_relative_pos.w;
	dir_light_relative_pos.xy = dir_light_relative_pos.xy * 0.5 + 0.5;
	float shadow = texture(sampler2DShadow(dir_light_shadow, shadow_sampler), dir_light_relative_pos.xyz);
	
	float diff_intensity = max(dot(normal, -light_direction), 0.0) * shadow;
	vec3 diffuse = diff_intensity * tex_diffuse;
	return diffuse;
}
// calculate directional light (sunlight/moonlight)
vec3 calc_dl(vec3 tex_diffuse, vec3 normal)
{
	vec3 light_direction = dir_light.direction;
	vec3 light_color = dir_light.color_intensity.xyz * dir_light.color_intensity.a;
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

