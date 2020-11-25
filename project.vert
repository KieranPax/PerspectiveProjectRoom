#version 330

layout (location = 0) in vec3 vPos;
layout (location = 1) in vec2 vTex;
layout (location = 2) in vec3 vNorm;

uniform mat4 viewMat;
uniform mat4 objMat;

out vec2 fromTex;
out vec2 toTex;

void main() {
    vec4 working_pos;

    working_pos = viewMat * objMat * vec4(vPos, 1.);

    fromTex = working_pos.xy/working_pos.w;
    toTex = vTex*2-1;

    gl_Position = vec4(toTex, min(working_pos.z-2,-1), 1.);
}