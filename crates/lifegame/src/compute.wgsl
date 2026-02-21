
// https://conwaylife.com/wiki/Rule_integer
@group(0) @binding(0) var<uniform> rule: u32;
@group(0) @binding(1) var input_texture: texture_storage_2d<rgba8uint, read>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8uint, write>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(input_texture);
  
  let t0 = textureLoad(input_texture, id.xy).r;
  let t1 = textureLoad(input_texture, (id.xy + vec2u(size.x + 0, size.y + 1)) % size).r;
  let t2 = textureLoad(input_texture, (id.xy + vec2u(size.x + 0, size.y - 1)) % size).r;
  let t3 = textureLoad(input_texture, (id.xy + vec2u(size.x - 1, size.y + 0)) % size).r;
  let t4 = textureLoad(input_texture, (id.xy + vec2u(size.x - 1, size.y + 1)) % size).r;
  let t5 = textureLoad(input_texture, (id.xy + vec2u(size.x - 1, size.y - 1)) % size).r;
  let t6 = textureLoad(input_texture, (id.xy + vec2u(size.x + 1, size.y + 0)) % size).r; 
  let t7 = textureLoad(input_texture, (id.xy + vec2u(size.x + 1, size.y + 1)) % size).r;
  let t8 = textureLoad(input_texture, (id.xy + vec2u(size.x + 1, size.y - 1)) % size).r;

  let sum = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;

  let next_life = ((rule >> sum) >> (t0 * 8)) & 1;

  textureStore(output_texture, id.xy, vec4u(next_life, 0, 0, 0));
}