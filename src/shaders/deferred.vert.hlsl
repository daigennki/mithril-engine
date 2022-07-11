cbuffer model : register(b0)
{
	float4x4 transform;	    // rotation, scale, and translation
};
cbuffer projviewmat : register(b0, space2)
{
    float4x4 projview;
};
struct VS_INPUT
{
    float3 position : POSITION;
    float2 uv : TEXCOORD;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float3 bitangent : BINORMAL;
};
struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float2 tex_coords : TEXCOORD;
    float3 tangent : TANGENT;
    float3 bitangent : BINORMAL;
    float3 normal : NORMAL;
};
VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    output.pos = mul(transform, float4(input.position, 1.0));
    output.pos = mul(projview, output.pos);
    output.tex_coords = input.uv;

    float3x3 transform3 = float3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    float3x3 tbn = float3x3(input.tangent, input.bitangent, input.normal);
    tbn = mul(transform3, tbn);
    output.tangent = tbn[0];
    output.bitangent = tbn[1];
    output.normal = tbn[2];
    return output;
}