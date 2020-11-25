#version 330

uniform sampler2D MainTexture;

in vec3 fPos;
in vec3 fNorm;
in vec2 fTex;

out vec4 FragColor;

void main() {
    FragColor = texture(MainTexture,fTex);
}
