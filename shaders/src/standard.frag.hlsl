/* Samplers/Textures */
Texture2D gDepth : register(t0, space1);
Texture2D gAlbedo : register(t1, space1);
Texture2D gNormalSpec : register(t2, space1);
SamplerComparisonState shadowSampler : register(s0);	// Only shadows require a sampler, others use `Load` with the texel coordinates
Texture2DArray dl_shadows : register(t5);
TextureCubeArray pl_shadows : register(t6);
Texture2DArray sl_shadows : register(t7);
Texture2D ssaoResult : register(t8);

/* UBOs */
cbuffer renderParams : register(b0)
{
	float4x4 proj;
	float4x4 view;
	float4x4 inverseProjView;
	float3 camera_pos;
	uint rpFiller;
	uint msCount;
	bool ssao_enabled;
    uint pointLightCount;
    uint spotLightCount;
};

/* Environment light */
struct DirLight 
{
    float3 direction;
	float filler1;
	float3 color;
	float filler2;
	float4x4 mats[4];
};
cbuffer dirLightUBO : register(b1)
{
	DirLight dirLight;
};

/* Point lights */
struct PointLight 
{
    float4 position;
    float4 direction;
    float3 color;
    float range;
    float lightLinear;
    float lightQuadratic;
    float inner_cutoff;
    float outer_cutoff;
    float4x4 mats[6];
    float4x4 filler;   // align to make this struct 512 bytes
};
StructuredBuffer<PointLight> pointLights : register(t11);

/* Spot lights */
struct SpotLight 
{
	float4 position;
    float4 direction;    // ignored
    float3 color;
    float range;
    float lightLinear;
    float lightQuadratic;
    float inner_cutoff;     // ignored
    float outer_cutoff;     // ignored
    float4x4 mats[1];
};
StructuredBuffer<SpotLight> spotLights : register(t12);

float3 calcDiffSpec(float3 lightDir, float3 normal, float3 viewDir, float3 tex_diffuse, float tex_specular)
{
	float diff_intensity = max(dot(normal, lightDir), 0.0);
    float3 diffuse = diff_intensity * tex_diffuse;

	float3 reflect_dir = reflect(-lightDir, normal);
	float spec_intensity = max(dot(viewDir, reflect_dir), 0.0);
	float specular = spec_intensity * tex_specular;

	return diffuse + specular;
}
// calculate directional light (sunlight/moonlight)
float3 calc_dl(float3 tex_diffuse, float tex_specular, float3 FragPos, float3 normal, float3 view_dir, float3 ambient)
{
	float3 color_out = calcDiffSpec(dirLight.direction, normal, view_dir, tex_diffuse, tex_specular);
	bool inRange = false;
	for (int i = 0; i < 4; ++i)
	{	// only evaluate against closest entry in cascaded shadows
        float4 lsPos = mul(dirLight.mats[i], float4(FragPos, 1.0));
		if ((lsPos.x >= -lsPos.w && lsPos.x <= lsPos.w && 
			lsPos.y >= -lsPos.w && lsPos.y <= lsPos.w && 
			lsPos.z >= -lsPos.w && lsPos.z <= lsPos.w)
		)
		{
			inRange = true;
			lsPos /= lsPos.w;
			lsPos.xy = mad(lsPos.xy, 0.5, 0.5);
			color_out *= dl_shadows.SampleCmp(shadowSampler, float3(lsPos.xy, i), lsPos.z);	// project shadow onto fragment
			break;
		}
	}
	color_out += ambient;
	//if (!inRange) color_out *= float3(1.25, 1.0, 1.0);	// debug feature to check if fragment is out of envlight range
	return dirLight.color * color_out;
}
// calculate point light
float3 calc_pl(int index, float3 tex_diffuse, float tex_specular, float3 FragPos, float3 normal, float3 view_dir, float3 ambient)
{
	PointLight light = pointLights[index];

	float3 fragToLight = FragPos - light.position.xyz;
	float dist = length(fragToLight);
	if (dist > light.range * 2.0) return float3(0.0, 0.0, 0.0);

	float3 light_dir = normalize(-fragToLight);
	float3 color_out = calcDiffSpec(light_dir, normal, view_dir, tex_diffuse, tex_specular);
	fragToLight.y *= -1;
	color_out *= pl_shadows.SampleCmp(shadowSampler, float4(fragToLight, index), dist / (light.range * 2));
	color_out += ambient;
	float attenuation = 1.0 / (1.0 + light.lightLinear * dist + light.lightQuadratic * dist * dist);
	color_out *= attenuation;
	return light.color * color_out;
}
// calculate spot light
float3 calc_sl(int index, float3 tex_diffuse, float tex_specular, float3 FragPos, float3 normal, float3 view_dir, float3 ambient)
{
	SpotLight light = spotLights[index];

	float3 fragToLight = light.position.xyz - FragPos;
	float dist = length(fragToLight);
	if (dist > light.range * 2.0) return float3(0.0, 0.0, 0.0);

    float4 lsPos = mul(light.mats[0], float4(FragPos, 1.0f));
	if ((lsPos.x >= -lsPos.w && lsPos.x <= lsPos.w && 
		lsPos.y >= -lsPos.w && lsPos.y <= lsPos.w && 
		lsPos.z >= -lsPos.w && lsPos.z <= lsPos.w)
	)
	{
		lsPos /= lsPos.w;
		lsPos.xy = mad(lsPos.xy, 0.5, 0.5);

		float3 light_dir = normalize(fragToLight);
		float3 color_out = calcDiffSpec(light_dir, normal, view_dir, tex_diffuse, tex_specular);
		color_out *= sl_shadows.SampleCmp(shadowSampler, float3(lsPos.xy, index), lsPos.z);
		color_out += ambient;

		// calculate intensity to simulate spot light cone
		float theta = dot(light_dir, light.direction.xyz) - light.outer_cutoff;
		float epsilon = light.inner_cutoff - light.outer_cutoff;
		float intensity = clamp(theta / epsilon, 0.0, 1.0);
		float attenuation = intensity / (1.0 + light.lightLinear * dist + light.lightQuadratic * dist * dist);
		color_out *= attenuation;
		return light.color * color_out;
	}
	return float3(0.0, 0.0, 0.0);
}

struct PS_OUTPUT
{
    float4 color : SV_Target;
    float depth : SV_Depth;
};
PS_OUTPUT main(float2 texcoords : TEXCOORD)
{
    // Obtain data from geometry framebuffers
	int2 texelPos;
	gAlbedo.GetDimensions(texelPos.x, texelPos.y);
	texelPos = int2(texelPos * texcoords);
	float Depth = gDepth.Load(int3(texelPos, 0)).r;
	float4 Albedo = gAlbedo.Load(int3(texelPos, 0));
	float4 NormalSpec = gNormalSpec.Load(int3(texelPos, 0));
	float3 Normal = NormalSpec.rgb;
	float Specular = NormalSpec.a;

	// smooth ambient occlusion, calculate ambient
	float ambOcclusion = 0.04;
	if (ssao_enabled)
	{
		ambOcclusion = 0.0;
		for (int x = -1; x <= 1; ++x)
		{
			for (int y = -1; y <= 1; ++y)
			{
				ambOcclusion += ssaoResult.Load(int3(texelPos + int2(x, y), 0)).r;
			}
		}
		ambOcclusion /= (3 * 3);
		ambOcclusion *= 0.04;
	}
	float3 Ambient = Albedo.xyz * ambOcclusion;

	// reconstruct world position from window space coordinates, depth, and inverse projection/view matrix
	float2 windowcoords = mad(texcoords, 2.0, -1.0);
	float4 ReconstructedPos = float4(windowcoords, Depth, 1.0);
	ReconstructedPos = mul(inverseProjView, ReconstructedPos);
	float3 FragPos = ReconstructedPos.xyz / ReconstructedPos.w;

	// then calculate lighting as usual
	float3 viewDir = normalize(camera_pos - FragPos);
	float3 shaded = calc_dl(Albedo.rgb, Specular, FragPos, Normal, viewDir, Ambient);
	int i = 0;
	for (i = 0; i < pointLightCount; ++i) shaded += calc_pl(i, Albedo.rgb, Specular, FragPos, Normal, viewDir, Ambient);
	for (i = 0; i < spotLightCount; ++i) shaded += calc_sl(i, Albedo.rgb, Specular, FragPos, Normal, viewDir, Ambient);

	// experimental tonemapping (TODO: use a separate shader?)
	// this gets disabled when HDR is enabled (passthrough to final output)
	float dynRange = 4.0;
	shaded /= dynRange;

    PS_OUTPUT output;
    output.color = float4(shaded, Albedo.a);
	output.depth = Depth;
    return output;
}
