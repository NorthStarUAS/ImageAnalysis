#version 130
 
const float cutoff = 3.0;
const float scalar = 1.0 / 3.0;

uniform sampler2D p3d_Texture0;
 
// Input from vertex shader
in vec2 texcoord;
in vec4 colorV;
out vec4 out_color;
 
void main() {
  vec4 color = texture(p3d_Texture0, texcoord);
  // simple passthrough
  // out_color = color.bgra * colorV.bgra;

  // Emphasize red/green objects
  out_color.r = min(color.r / max(color.g, 0.01), cutoff) * scalar * colorV.r;
  out_color.g = min(color.g / max(color.r, 0.01), cutoff) * scalar * colorV.g;
  out_color.b = 0;
  out_color.a = color.a * colorV.a;

  // "Visual" index favoring green
  // float vari = (color.g - color.r) / (color.g + color.r);
  // out_color.r = clamp((-0.5*vari)+0.5, 0.0, 1.0) * colorV.r;
  // out_color.g = clamp((0.5*vari)+0.5, 0.0, 1.0) * colorV.g;
  // out_color.b = 0;
  // out_color.a = color.a * colorV.a;

  // "Visual" index favoring red
  // float vari = (color.r - color.g) / (color.r + color.g);
  // out_color.r = clamp((0.5*vari)+0.5, 0.0, 1.0) * colorV.r;
  // out_color.g = clamp((-0.5*vari)+0.5, 0.0, 1.0) * colorV.g;
  // out_color.b = 0;
  // out_color.a = color.a * colorV.a;
}
