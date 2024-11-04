#version 450

void main()
{
    vec2 p = vec2(0.0);
    float s = 1.0;
    
    uint idx = gl_VertexIndex;
    for (uint i = 0u; i < 16u; i++)
    {
        uint nidx = idx / 3u;
        uint m = idx - 3u * nidx;
        idx = nidx;
        
        vec2 o = m == 0 ? vec2(0.0, 0.5) : m == 1 ? vec2(-0.5, -0.5) : vec2(0.5, -0.5);
        p += s * o;
        s *= 0.5;
    }

    gl_Position = vec4(p, 0.5, 1.0);
    gl_PointSize = 1.0;
}
