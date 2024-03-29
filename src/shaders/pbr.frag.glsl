#version 460
#include "materials_common.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 world_pos;
layout(location = 3) flat in int instance_index;

#define TEXTURE_INDEX_OFFSET_BASE_COLOR 0

vec3 calc_diff(vec3 light_direction, vec3 normal, vec3 tex_diffuse)
{
	// Go through all of the shadow maps, and use the closest shadow map for which the point is in view.
	// If the point is not in view of any of them, fall back to basic shading without shadow mapping.
	// We go from furthest to closest here so that the closest shadow map takes priority over others 
	// if the position is inside of it.
	float shadow_actual = 1.0;
	for (int i = CSM_COUNT-1; i >= 0; i -= 1) {
		vec4 dir_light_relative_pos = dir_light.projviews[i] * vec4(world_pos, 1.0);
		dir_light_relative_pos /= dir_light_relative_pos.w;
		dir_light_relative_pos.xy = dir_light_relative_pos.xy * 0.5 + 0.5;

		bool in_range =
			dir_light_relative_pos.x >= 0.0 && dir_light_relative_pos.x < 1.0
			&& dir_light_relative_pos.y >= 0.0 && dir_light_relative_pos.y < 1.0
			&& dir_light_relative_pos.z >= 0.0 && dir_light_relative_pos.z < 1.0;

		vec4 shadow_coord = vec4(dir_light_relative_pos.xy, i, dir_light_relative_pos.z);
		float shadow = texture(sampler2DArrayShadow(dir_light_shadow, shadow_sampler), shadow_coord);
		shadow_actual = (in_range ? shadow : shadow_actual);
	}

	float diff_intensity = max(dot(normal, -light_direction), 0.0) * shadow_actual;
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
	int base_color_index = instance_index + TEXTURE_INDEX_OFFSET_BASE_COLOR;
	vec4 tex_color = texture(sampler2D(textures[base_color_index], sampler0), texcoord);

	if (TRANSPARENCY_PASS) {
		tex_color.rgb *= tex_color.a;
	}

	vec3 shaded = calc_dl(tex_color.rgb, normal);

	write_pixel(vec4(shaded, tex_color.a));
}

