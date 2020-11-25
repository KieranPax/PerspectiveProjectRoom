#version 330

layout (location = 0) in vec3 vPos;
layout (location = 1) in vec2 vTex;
layout (location = 2) in vec3 vNorm;

uniform mat4 viewMat;
uniform mat4 objMat;

out vec3 fPos;
out vec3 fNorm;
out vec2 fTex;

void main() {
    vec4 working_pos;

    working_pos = objMat*vec4(vPos, 1.);
    gl_Position = viewMat*working_pos;
    fPos = working_pos.xyz;

    fNorm = (objMat * vec4(vNorm,0.)).xyz;

    fTex=vTex;
}