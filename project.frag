#version 330

in vec2 fromTex;
in vec2 toTex;

uniform sampler2D MainTexture;

out vec4 FragColor;

void main() {
    if (fromTex.x > 1 || fromTex.x < -1) discard;
    if (fromTex.y > 1 || fromTex.y < -1) discard;
    FragColor = texture(MainTexture, (fromTex-1)/2);
    if (FragColor.w < 0.01) discard;
}
