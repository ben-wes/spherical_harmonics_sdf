// Ben Wesch, 2024

// huge parts stolen from https://www.shadertoy.com/view/tlcSzr
// and https://www.shadertoy.com/view/XtcGWn
// harmonics simplified according to https://en.wikipedia.org/wiki/Table_of_spherical_harmonics ... unfortunately ending at 4th degree

#define PI 3.1415926

#define MAX_MARCHING_STEPS 200
#define MIN_DIST  0.
#define MAX_DIST 70.
#define EPSILON 0.001

// uniforms sent from Pd
uniform vec2 resolution;
uniform vec2 mouse_drag;
uniform float time;
uniform int mode;
uniform float                      c0; // harmonic coefficients
uniform float                 c1,  c2,  c3;
uniform float            c4,  c5,  c6,  c7,  c8;
uniform float       c9, c10, c11, c12, c13, c14, c15;
uniform float c16, c17, c18, c19, c20, c21, c22, c23, c24;

float get_spherical_harmonics(
    vec3 v,
                                                    float f00,
                                        float f1n1, float f10, float f11, 
                            float f2n2, float f2n1, float f20, float f21, float f22, 
                float f3n3, float f3n2, float f3n1, float f30, float f31, float f32, float f33,
    float f4n4, float f4n3, float f4n2, float f4n1, float f40, float f41, float f42, float f43, float f44
){

    float x = v.x;
    float y = v.y;
    float z = v.z;

    float x2 = x*x;
    float y2 = y*y;
    float z2 = z*z;

    float xy = x*y;
    float yz = y*z;
    float zx = z*x;

    float xyz = x*y*z;

    return
          f00  * .5*sqrt(1./PI)  

        + f1n1 * sqrt(.75/PI) * y  
        + f10  * sqrt(.75/PI) * z  
        + f11  * sqrt(.75/PI) * x 

        + f2n2 * .50 * sqrt(15./PI) * xy 
        + f2n1 * .50 * sqrt(15./PI) * yz 
        + f20  * .25 * sqrt( 5./PI) * (3.*z2-1.)
        + f21  * .50 * sqrt(15./PI) * zx 
        + f22  * .25 * sqrt(15./PI) * (x2-y2)  

        + f3n3 * .25 * sqrt( 35./(2.*PI)) * y*(3.*x2-y2)
        + f3n2 * .50 * sqrt(105./    PI)  * xyz
        + f3n1 * .25 * sqrt( 21./(2.*PI)) * y*(5.*z2-1.)
        + f30  * .25 * sqrt(  7./    PI)  * z*(5.*z2-3.)
        + f31  * .25 * sqrt( 21./(2.*PI)) * x*(5.*z2-1.)
        + f32  * .25 * sqrt(105./    PI)  * z*(x2-y2)
        + f33  * .25 * sqrt( 35./(2.*PI)) * x*(x2-3.*y2)

        + f4n4 * .75   * sqrt(35./(   PI)) * xy*(x2-y2) 
        + f4n3 * .75   * sqrt(35./(2.*PI)) * yz*(3.*x2-y2)
        + f4n2 * .75   * sqrt( 5./(   PI)) * xy*(7.*z2-1.)
        + f4n1 * .75   * sqrt( 5./(2.*PI)) * yz*(7.*z2-3.)
        + f40  * .09375 * (35.*z2*z2-30.*z2+3.)
        + f41  * .75   * sqrt( 5./(2.*PI)) * zx*(7.*z2-3.)
        + f42  * .375  * sqrt( 5./(   PI)) * (x2-y2)*(7.*z2-1.)
        + f43  * .75   * sqrt(35./(2.*PI)) * zx*(x2-3.*y2)
        + f44  * .1875 * sqrt(35./(   PI)) * (x2*(x2-3.*y2) - y2*(3.*x2-y2))
      ;
}

// "get_distance_from_point_to_spherical_harmonics_blob" returns the 
// signed distance of a point to the surface of a spheroid whose surface is 
// offset using a linear combination of spherical harmonics. 

// A0 point position
// B0 blob origin
// r  blob reference radius
//   the radius of a sphere where f00==1 and f1n1..f22 == 0
// f00..f22 blob expansion coefficients
//   the expansion coefficients to the spherical harmonics series
//   that describe the radius of a blob at a given set of lat long coordinates

vec2 getHarmonicPosition(int l, int m) {
    float x = float(m) * 0.1 + 0.5;
    float y = 0.8 - float(l) * 0.15; // Assuming max l is 4
    return vec2(x, y);
}

vec3 getHarmonicColor(float value) {
    if (value > 0.0) {
        return vec3(1.0 + value * 0.5, 1.0 - value * 0.2, 1.0 - value * 0.6);
    } else {
        return vec3(1.0 + value * 0.5, 1.0 + value * 0.2, 1.0 - value * 0.6);
    }
}

float getHarmonicValue(int l, int m) {
    if (l == 0 && m == 0) return c0;
    if (l == 1) {
        if (m == -1) return c1;
        if (m == 0)  return c2;
        if (m == 1)  return c3;
    }
    if (l == 2) {
        if (m == -2) return c4;
        if (m == -1) return c5;
        if (m == 0)  return c6;
        if (m == 1)  return c7;
        if (m == 2)  return c8;
    }
    if (l == 3) {
        if (m == -3) return c9;
        if (m == -2) return c10;
        if (m == -1) return c11;
        if (m == 0)  return c12;
        if (m == 1)  return c13;
        if (m == 2)  return c14;
        if (m == 3)  return c15;
    }
    if (l == 4) {
        if (m == -4) return c16;
        if (m == -3) return c17;
        if (m == -2) return c18;
        if (m == -1) return c19;
        if (m == 0)  return c20;
        if (m == 1)  return c21;
        if (m == 2)  return c22;
        if (m == 3)  return c23;
        if (m == 4)  return c24;
    }
    return 0.0;
}

float get_radius_of_blob_towards_point(vec3 D) {
    vec3 Dhat = normalize(D);
    float fijYij = get_spherical_harmonics(
        Dhat,
                             c0,
                        c1,  c2,  c3,
                   c4,  c5,  c6,  c7,  c8,
              c9, c10, c11, c12, c13, c14, c15,
        c16, c17, c18, c19, c20, c21, c22, c23, c24
    );
    return fijYij;
}

float sceneSDF(vec3 p) {
    return length(p) - .1 - 2. * abs(get_radius_of_blob_towards_point(p));
}

/**
 * Return the shortest distance from the eyepoint to the scene surface along
 * the marching direction. If no part of the surface is found between start and end,
 * return end.
 * 
 * eye: the eye point, acting as the origin of the ray
 * marchingDirection: the normalized direction to march in
 * start: the starting distance away from the eye
 * end: the max distance away from the ey to march before giving up
 */
float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    float lastDist = MAX_DIST;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(eye + depth * marchingDirection);
        if (dist < EPSILON) {
            return depth;
        }
        // Adaptive step size with dampening
        float step = min(abs(dist) * 0.5, lastDist * 0.5);
        depth += max(step, EPSILON * depth);
        
        // Early termination
        if (depth >= end) {
            return end;
        }
        lastDist = dist;
    }
    return end;
}

/**
 * Return the normalized direction to march in from the eye point for a single pixel.
 * 
 * fieldOfView: vertical field of view in degrees
 * size: resolution of the output image
 * fragCoord: the x,y coordinate of the pixel in the output image
 */
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

/**
 * Using the gradient of the SDF, estimate the normal on the surface at point p.
 */
vec3 estimateNormal(vec3 p) {
    float h = 0.001;
    vec2 k = vec2(1, -1);
    return normalize(
        k.xyy * sceneSDF(p + k.xyy * h) +
        k.yyx * sceneSDF(p + k.yyx * h) +
        k.yxy * sceneSDF(p + k.yxy * h) +
        k.xxx * sceneSDF(p + k.xxx * h)
    );
}

/**
 * Lighting contribution of a single point light source via Phong illumination.
 * 
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = reflect(-L, N);
    
    float dotLN = max(dot(L, N), 0.0);
    float dotRV = max(dot(R, V), 0.0);
    
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

/**
 * Lighting via Phong illumination.
 * 
 * The vec3 returned is the RGB color of that point after lighting is applied.
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = 0.4 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light1Pos = vec3(6.0 * sin(0.2 * time),
                          4.0 * cos(0.2 * time),
                          2.0);
    vec3 light1Intensity = vec3(0.7);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light1Pos,
                                  light1Intensity);
    
    vec3 light2Pos = vec3(1.0 * sin(0.077 * time),
                          -1.0 * cos(0.077 * time),
                          3.0);
    vec3 light2Intensity = vec3(0.3);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light2Pos,
                                  light2Intensity);    
    return color;
}

/**
 * Return a transform matrix that will transform a ray from view space
 * to world coordinates, given the eye point, the camera target, and an up vector.
 *
 * This assumes that the center of the camera is aligned with the negative z axis in
 * view space when calculating the ray marching direction. See rayDirection.
 */
mat4 viewMatrix(vec3 eye, vec3 center, vec3 up) {
    // Based on gluLookAt man page
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat4(
        vec4(s, 0.0),
        vec4(u, 0.0),
        vec4(-f, 0.0),
        vec4(0.0, 0.0, 0.0, 1)
    );
}

// Rotation matrix around the X axis.
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

// Rotation matrix around the Y axis.
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

// Rotation matrix around the Z axis.
mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}

vec3 drawGrid(vec2 uv) {
    return vec3(max(.1, .92 - .88 * min(
        pow(mod(uv.x * 10., 1.), 0.02),
        pow(mod(uv.y * 10., 1.), 0.02)
    )));
}

vec2 singleHarmonicShape(vec3 p, int l, int m, float amplitude) {
    vec3 rotatedP = rotateX(PI/2.0) * p;
    vec3 dir = normalize(rotatedP);
    float r = length(p);
    float shape = get_spherical_harmonics(
        dir,
        l == 0 && m == 0 ? 1.0 : 0.0,
        l == 1 && m == -1 ? 1.0 : 0.0, l == 1 && m == 0 ? 1.0 : 0.0, l == 1 && m == 1 ? 1.0 : 0.0,
        l == 2 && m == -2 ? 1.0 : 0.0, l == 2 && m == -1 ? 1.0 : 0.0, l == 2 && m == 0 ? 1.0 : 0.0, l == 2 && m == 1 ? 1.0 : 0.0, l == 2 && m == 2 ? 1.0 : 0.0,
        l == 3 && m == -3 ? 1.0 : 0.0, l == 3 && m == -2 ? 1.0 : 0.0, l == 3 && m == -1 ? 1.0 : 0.0, l == 3 && m == 0 ? 1.0 : 0.0, l == 3 && m == 1 ? 1.0 : 0.0, l == 3 && m == 2 ? 1.0 : 0.0, l == 3 && m == 3 ? 1.0 : 0.0,
        l == 4 && m == -4 ? 1.0 : 0.0, l == 4 && m == -3 ? 1.0 : 0.0, l == 4 && m == -2 ? 1.0 : 0.0, l == 4 && m == -1 ? 1.0 : 0.0, l == 4 && m == 0 ? 1.0 : 0.0, l == 4 && m == 1 ? 1.0 : 0.0, l == 4 && m == 2 ? 1.0 : 0.0, l == 4 && m == 3 ? 1.0 : 0.0, l == 4 && m == 4 ? 1.0 : 0.0
    );
    float dist = r - abs(amplitude * shape);
    return vec2(dist, shape * sign(amplitude));
}

float getHarmonicValue(vec3 p, int l, int m) {
    vec3 rotatedP = rotateX(PI/2.0) * p;
    vec3 dir = normalize(rotatedP);
    return get_spherical_harmonics(
        dir,
        l == 0 && m == 0 ? 1.0 : 0.0,
        l == 1 && m == -1 ? 1.0 : 0.0, l == 1 && m == 0 ? 1.0 : 0.0, l == 1 && m == 1 ? 1.0 : 0.0,
        l == 2 && m == -2 ? 1.0 : 0.0, l == 2 && m == -1 ? 1.0 : 0.0, l == 2 && m == 0 ? 1.0 : 0.0, l == 2 && m == 1 ? 1.0 : 0.0, l == 2 && m == 2 ? 1.0 : 0.0,
        l == 3 && m == -3 ? 1.0 : 0.0, l == 3 && m == -2 ? 1.0 : 0.0, l == 3 && m == -1 ? 1.0 : 0.0, l == 3 && m == 0 ? 1.0 : 0.0, l == 3 && m == 1 ? 1.0 : 0.0, l == 3 && m == 2 ? 1.0 : 0.0, l == 3 && m == 3 ? 1.0 : 0.0,
        l == 4 && m == -4 ? 1.0 : 0.0, l == 4 && m == -3 ? 1.0 : 0.0, l == 4 && m == -2 ? 1.0 : 0.0, l == 4 && m == -1 ? 1.0 : 0.0, l == 4 && m == 0 ? 1.0 : 0.0, l == 4 && m == 1 ? 1.0 : 0.0, l == 4 && m == 2 ? 1.0 : 0.0, l == 4 && m == 3 ? 1.0 : 0.0, l == 4 && m == 4 ? 1.0 : 0.0
    );
}

vec3 getColorFromRadiusAndPole(float radius, float poleValue, float maxRadius) {
    vec3 positiveColor = vec3(2.0, 0.2, 0.0);
    vec3 negativeColor = vec3(0.0, 0.9, 2.0);
    vec3 neutralColor = vec3(0.8, 0.8, 0.8);
    
    vec3 color = mix(neutralColor, poleValue > 0.0 ? positiveColor : negativeColor, pow(radius * 2., 1.4));
    return color;
}

void main()
{
    vec2 uv = gl_FragCoord.xy / resolution;

    if (mode == 0) {
        vec3 viewDir = rayDirection(48.0, resolution.xy, gl_FragCoord.xy);
        vec3 eye = vec3(0.0, -24.0, 0.0) * rotateX(mouse_drag.y) * rotateZ(mouse_drag.x);
        
        mat4 viewToWorld = viewMatrix(eye, vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
        
        vec3 worldDir = (viewToWorld * vec4(viewDir, 0.0)).xyz;
        
        float dist = shortestDistanceToSurface(eye, worldDir, MIN_DIST, MAX_DIST);
        
        if (dist < MAX_DIST - EPSILON) {
            vec3 p = eye + dist * worldDir;
            vec3 K_a = vec3(0.4);
            vec3 K_d = vec3(0.4);
            vec3 K_s = vec3(0.04);
            float shininess = 10.0;
            
            float radius = get_radius_of_blob_towards_point(p);
            vec3 color = phongIllumination(K_a, K_d, K_s, shininess, p, eye);
            gl_FragColor = vec4(color.r + radius * 0.5, color.g - radius * 0.2, color.b - radius * 0.6, 1.0);
            return;
        }
    }
    else if (mode == 1) {
        vec3 lightDir = normalize(vec3(1.0 + sin(time * 0.5) * 0.8, 1.0, -1.0));
        mat3 rot = rotateY(mouse_drag.x) * rotateX(mouse_drag.y);

        for (int l = 0; l <= 4; l++) {
            for (int m = -l; m <= l; m++) {
                vec2 pos = getHarmonicPosition(l, m);
                float size = 0.18;
                vec2 boxMin = pos - size * 0.5;
                vec2 boxMax = pos + size * 0.5;
                
                if (uv.x >= boxMin.x && uv.x <= boxMax.x && uv.y >= boxMin.y && uv.y <= boxMax.y) {
                    float amplitude = getHarmonicValue(l, m);
                    vec2 localUV = (uv - boxMin) / size - 0.5;
                    vec3 ro = vec3(localUV * 2.0, -3.0);
                    vec3 rd = normalize(vec3(0.0, 0.0, 1.0));
                    
                    float t = 0.0;
                    for(int i = 0; i < 32; i++) {
                        vec3 p = ro + rd * t;
                        float d = singleHarmonicShape(rot * p, l, m, amplitude).x;
                        if(d < EPSILON) {
                            vec3 normal = normalize(vec3(
                                singleHarmonicShape(rot * (p + vec3(0.001, 0.0, 0.0)), l, m, amplitude).x - singleHarmonicShape(rot * (p - vec3(0.001, 0.0, 0.0)), l, m, amplitude).x,
                                singleHarmonicShape(rot * (p + vec3(0.0, 0.001, 0.0)), l, m, amplitude).x - singleHarmonicShape(rot * (p - vec3(0.0, 0.001, 0.0)), l, m, amplitude).x,
                                singleHarmonicShape(rot * (p + vec3(0.0, 0.0, 0.001)), l, m, amplitude).x - singleHarmonicShape(rot * (p - vec3(0.0, 0.0, 0.001)), l, m, amplitude).x
                            ));
                            float diff = max(dot(normal, lightDir), 0.0);
                            
                            float radius = length(p);
                            vec3 shapeColor = getColorFromRadiusAndPole(radius, singleHarmonicShape(rot * p, l, m, amplitude).y, 1.0 + abs(amplitude));
                            gl_FragColor = vec4(mix(shapeColor * 0.3, shapeColor, diff), 1.0);
                            return;
                        }
                        t += max(abs(d) * 0.5, 0.01);
                        if(t > 5.0) break;
                    }
                }
            }
        }
    }

    gl_FragColor = vec4(drawGrid(uv), 1.0);
}
