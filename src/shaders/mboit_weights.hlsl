 
// The shader code used for moment-based OIT moment weight calculation (stage 3), meant to be included by other shaders.

/* sum(rgb * a, a) */
//Texture2D moments_in : register(t0, space3);
[[vk::input_attachment_index(0)]] SubpassInput moments_in : register(t0, space3);

/* prod(1 - a) */
//Texture2D optical_depth_in : register(t1, space3);
[[vk::input_attachment_index(1)]] SubpassInput optical_depth_in : register(t1, space3);

/* minimum depth for correction */
[[vk::input_attachment_index(2)]] SubpassInput min_depth : register(t2, space3);

/*cbuffer tex_dim : register(b2, space3)
{
	uint2 texture_dimensions;
};*/

struct PS_OUTPUT
{
	float4 accum : SV_Target0;
	float revealage : SV_Target1;
};

// this function mostly copied from Shaders/Shadow.fx available in the demo code at https://jcgt.org/published/0006/01/03/
/*! Given a sampled value from a four-moment shadow map and a computed shadow map 
   depth for a point at the same location this function outputs 1.0, if the fragment 
   is in shadow 0.0f, if the fragment is lit and an intermediate value for partial 
   shadow. The returned value is an optimal lower bound except for the fact that it 
   does not exploit the knowledge that the original distribution has support in 
   [0,1].*/
void Compute4MomentUnboundedShadowIntensity(out float OutShadowIntensity,
    float4 Biased4Moments,float FragmentDepth,float DepthBias)
{
	// Use short-hands for the many formulae to come
	float4 b=Biased4Moments;
	float3 z;
	z[0]=FragmentDepth-DepthBias;

	// Compute a Cholesky factorization of the Hankel matrix B storing only non-
	// trivial entries or related products
	float L21D11=mad(-b[0],b[1],b[2]);
	float D11=mad(-b[0],b[0], b[1]);
	float SquaredDepthVariance=mad(-b[1],b[1], b[3]);
	float D22D11=dot(float2(SquaredDepthVariance,-L21D11),float2(D11,L21D11));
	float InvD11=1.0f/D11;
	float L21=L21D11*InvD11;
	float D22=D22D11*InvD11;
	float InvD22=1.0f/D22;

	// Obtain a scaled inverse image of bz=(1,z[0],z[0]*z[0])^T
	float3 c=float3(1.0f,z[0],z[0]*z[0]);
	// Forward substitution to solve L*c1=bz
	c[1]-=b.x;
	c[2]-=b.y+L21*c[1];
	// Scaling to solve D*c2=c1
	c[1]*=InvD11;
	c[2]*=InvD22;
	// Backward substitution to solve L^T*c3=c2
	c[1]-=L21*c[2];
	c[0]-=dot(c.yz,b.xy);
	// Solve the quadratic equation c[0]+c[1]*z+c[2]*z^2 to obtain solutions 
	// z[1] and z[2]
	float InvC2=1.0f/c[2];
	float p=c[1]*InvC2;
	float q=c[0]*InvC2;
	float D=(p*p*0.25f)-q;
	float r=sqrt(D);
	z[1]=-p*0.5f-r;
	z[2]=-p*0.5f+r;
	// Compute the shadow intensity by summing the appropriate weights
	float4 Switch=
		(z[2]<z[0])?float4(z[1],z[0],1.0f,1.0f):(
		(z[1]<z[0])?float4(z[0],z[1],0.0f,1.0f):
		float4(0.0f,0.0f,0.0f,0.0f));
	float Quotient=(Switch[0]*z[2]-b[0]*(Switch[0]+z[2])+b[1])/((z[2]-Switch[1])*(z[0]-z[1]));
	OutShadowIntensity=Switch[2]+Switch[3]*Quotient;
	OutShadowIntensity=saturate(OutShadowIntensity);
}

float depth_to_unit(float z, float c0, float c1)
{
	return log(z * c0) * c1;
}
float hamburger4msm(float4 moments, float z)
{
	moments = lerp(moments, float4(0.0, 0.375, 0.0, 0.375), 3.0e-7);
	float result;
	Compute4MomentUnboundedShadowIntensity(result, moments, z, 0.0);
	return result;
}
float calc_w(float z, float alpha, float2 screen_pos)
{
	// TODO: use a descriptor set here to reflect changes to the camera near/far planes
	const float near = 0.25;
	const float far = 5000.0;
	const float c0 = 1.0 / near;
	const float c1 = 1.0 / log(far / near);

	//float2 texcoords = mad(screen_pos, 0.5, 0.5);
	//int2 tex_coords_int = int2(texcoords * texture_dimensions);
    //float4 moments = moments_in.Load(int3(tex_coords_int, 0));
	//float total_od = optical_depth_in.Load(int3(tex_coords_int, 0));
	float4 moments = moments_in.SubpassLoad();
	float total_od = optical_depth_in.SubpassLoad();
	float unit_pos = depth_to_unit(z, c0, c1);

	if (total_od != 0.0) {
		moments /= total_od; // normalize
	}

	float ma = hamburger4msm(moments, unit_pos);
	ma = exp(-ma * total_od);
	float w = ma * alpha;

	// if this is *not* the top fragment, but the depth of this fragment is still close to the minimum depth,
	// correct the "gradient" that might appear under such conditions
	float min_z = min_depth.SubpassLoad();
	const float correction_threshold = 0.01;
	const float correction_inv = (1.0 / correction_threshold);
	if (z > min_z) {
		float z_diff = z - min_z;
		float correction = z_diff * correction_inv;
		w *= correction;
	}
	
	return w;
}
PS_OUTPUT write_transparent_pixel(float4 premul_reflect, float depth, float2 screen_pos) 
{ 
	float w = calc_w(depth, premul_reflect.a, screen_pos);
	
	PS_OUTPUT output;
	output.accum = premul_reflect * w;
    output.revealage = premul_reflect.a;
	return output;
}

