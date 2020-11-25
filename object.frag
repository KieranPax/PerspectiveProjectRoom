#version 330

in vec3 fPos;
in vec3 fNorm;
in vec2 fTex;

out vec4 FragColor;

void main() {
    FragColor = vec4((dot(fNorm, vec3(0, 1, 0))+0.5)*vec3(1., .4, 0.), 1.);
}
