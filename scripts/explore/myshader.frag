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

  // Emphasize dominant red vs. green extremes
  // float red = color.r / max(color.g, 0.01);
  // float green = color.g / max(color.r, 0.01);
  // out_color.r = min(red * red, cutoff) * scalar * colorV.r;
  // out_color.g = min(green * green, cutoff) * scalar * colorV.g;
  // out_color.b = 0;
  // out_color.a = color.a * colorV.a;

  // Emphasize red extremes (v1)
  float red = color.r / max(color.g, 0.01); // protect against divide-by-zero
  float green = color.g / max(color.r, 0.01);
  float lum = 0.21*color.r + 0.72*color.g + 0.07*color.b; // std rgb->gray
  float lum_factor = smoothstep(0.0, 0.2, lum); // knock out basement noise
  
  out_color.r = smoothstep(0.9, 3.0, red*lum_factor) * colorV.r;
  out_color.g = smoothstep(0.5, 2.5, green) * colorV.g;
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
