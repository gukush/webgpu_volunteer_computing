#version 300 es
precision highp float;

// Fullscreen triangle from gl_VertexID (WebGL2)
void main() {
  // verts: ( -1,-1 ), ( -1, 3 ), ( 3,-1 )
  vec2 pos = vec2(
    (gl_VertexID == 2) ? 3.0 : -1.0,
    (gl_VertexID == 1) ? 3.0 : -1.0
  );
  gl_Position = vec4(pos, 0.0, 1.0);
}
