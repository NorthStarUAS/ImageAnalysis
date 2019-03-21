#version 130
 
// Uniform inputs
uniform mat4 p3d_ModelViewProjectionMatrix;
 
// Vertex inputs
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
in vec4 p3d_Color;
 
// Output to fragment shader
out vec2 texcoord;
out vec4 colorV;
 
void main() {
  gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
  texcoord = p3d_MultiTexCoord0;
  colorV = p3d_Color;
}
