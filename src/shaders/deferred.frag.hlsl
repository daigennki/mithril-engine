struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float2 tex_coords : TEXCOORD;
    float3 tangent : TANGENT;
    float3 bitangent : BINORMAL;
    float3 normal : NORMAL;
};

struct PS_OUTPUT
{
    float4 gAlbedo : SV_Target0;
    float4 gNormalSpec : SV_Target1;
};
SamplerState sampler0 : register(s0, space1);
Texture2D diffuse_map : register(t0, space1);
Texture2D normal_map : register(t1, space1);
Texture2D specular_map : register(t2, space1);
cbuffer material : register(b0, space1)
{
    int translucent;
	bool diffuse_enabled;
	bool normal_enabled;
	bool specular_enabled;
};

PS_OUTPUT main(PS_INPUT input)
{
    // diffuse
    float4 diffuse = diffuse_map.Sample(sampler0, input.tex_coords);
#ifdef TRANSLUCENT_PASS
#else
    if (diffuse.a > 0.0) diffuse.a = 1.0;   // make opaque if this isn't a translucent material, unless the pixel is completely transparent
#endif
    if (diffuse.a == 0.0) discard;
    diffuse.rgb *= diffuse.a;   // premultiply alpha
    
    // normal and specular
    float3 normal = float3(0.0, 0.0, 1.0);
    if (normal_enabled)
    {
       normal.xy = normal_map.Sample(sampler0, input.tex_coords).xy;
       normal.xy = mad(normal.xy, 2.0, -1.0);
       normal.z = sqrt(saturate(1.0 - dot(normal.xy, normal.xy)));
    }
    float specular = specular_enabled ? specular_map.Sample(sampler0, input.tex_coords).r : 0.0;

    PS_OUTPUT output;
    output.gAlbedo = diffuse;
    float3x3 tbn = float3x3(input.tangent, input.bitangent, input.normal);
    output.gNormalSpec = float4(mul(tbn, normal), specular);
    return output;
}