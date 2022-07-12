cbuffer color : register(b0,space2)
{
	float4 color;
}
float4 main() : SV_Target
{
	return color;
}

