
// The shader code used for moment-based OIT moment weight calculation (stage 3), meant to be included by other shaders.

layout(set = 1, binding = 0) uniform sampler oit_sampler;

/* sum(rgb * a, a) */
layout(set = 1, binding = 1) uniform texture2D moments_in;

/* prod(1 - a) */
layout(set = 1, binding = 2) uniform texture2D optical_depth_in;

/* minimum depth for correction */
layout(set = 1, binding = 3) uniform texture2D min_depth;

layout(location = 0) out vec4 accum;
layout(location = 1) out float revealage;

// this function mostly copied from Shaders/Shadow.fx available in the demo code at https://jcgt.org/published/0006/01/03/
/*! Given a sampled value from a four-moment shadow map and a computed shadow map 
   depth for a point at the same location this function outputs 1.0, if the fragment 
   is in shadow 0.0f, if the fragment is lit and an intermediate value for partial 
   shadow. The returned value is an optimal lower bound except for the fact that it 
   does not exploit the knowledge that the original distribution has support in 
   [0,1].*/
void Compute4MomentUnboundedShadowIntensity(out float OutShadowIntensity,
    vec4 Biased4Moments,float FragmentDepth,float DepthBias)
{
	// Use short-hands for the many formulae to come
	vec4 b=Biased4Moments;
	vec3 z;
	z[0]=FragmentDepth-DepthBias;

	// Compute a Cholesky factorization of the Hankel matrix B storing only non-
	// trivial entries or related products
	float L21D11= -b[0] * b[1] + b[2];
	float D11= -b[0] * b[0] + b[1];
	float SquaredDepthVariance= -b[1] * b[1] + b[3];
	float D22D11=dot(vec2(SquaredDepthVariance,-L21D11),vec2(D11,L21D11));
	float InvD11=1.0f/D11;
	float L21=L21D11*InvD11;
	float D22=D22D11*InvD11;
	float InvD22=1.0f/D22;

	// Obtain a scaled inverse image of bz=(1,z[0],z[0]*z[0])^T
	vec3 c=vec3(1.0f,z[0],z[0]*z[0]);
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
	vec4 Switch=
		(z[2]<z[0])?vec4(z[1],z[0],1.0f,1.0f):(
		(z[1]<z[0])?vec4(z[0],z[1],0.0f,1.0f):
		vec4(0.0f,0.0f,0.0f,0.0f));
	float Quotient=(Switch[0]*z[2]-b[0]*(Switch[0]+z[2])+b[1])/((z[2]-Switch[1])*(z[0]-z[1]));
	OutShadowIntensity=Switch[2]+Switch[3]*Quotient;
	OutShadowIntensity=clamp(OutShadowIntensity, 0.0, 1.0);
}

float depth_to_unit(float z, float c0, float c1)
{
	return log(z * c0) * c1;
}
float hamburger4msm(vec4 moments, float z)
{
	moments = mix(moments, vec4(0.0, 0.375, 0.0, 0.375), 3.0e-7);
	float result;
	Compute4MomentUnboundedShadowIntensity(result, moments, z, 0.0);
	return result;
}
float calc_w(float alpha)
{
	float z = gl_FragCoord.z;

	// TODO: use a descriptor set here to reflect changes to the camera near/far planes
	const float near = 0.25;
	const float far = 5000.0;
	const float c0 = 1.0 / near;
	const float c1 = 1.0 / log(far / near);

	vec2 texcoord = gl_FragCoord.xy * 0.5 + 0.5;

	vec4 moments = texture(sampler2D(moments_in, oit_sampler), texcoord);
	float total_od = texture(sampler2D(optical_depth_in, oit_sampler), texcoord).r;
	float unit_pos = depth_to_unit(z, c0, c1);

	if (total_od != 0.0) {
		moments /= total_od; // normalize
	}

	float ma = hamburger4msm(moments, unit_pos);
	ma = exp(-ma * total_od);
	float w = ma * alpha;

	// if this is *not* the top fragment, but the depth of this fragment is still close to the minimum depth,
	// correct the "gradient" that might appear under such conditions
	float min_z = texture(sampler2D(min_depth, oit_sampler), texcoord).r;
	const float correction_factor = 100.0;
	if (z > min_z) {
		//w *= clamp((z - min_z) * correction_factor, 0.0, 1.0);
	}
	
	return w;
}
void write_transparent_pixel(vec4 premul_reflect) 
{ 
	float w = calc_w(premul_reflect.a);

	accum = premul_reflect * w;
    revealage = premul_reflect.a;
}

