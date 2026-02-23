struct Params {
  world_size: vec2u,
  // https://conwaylife.com/wiki/Rule_integer
  rule: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_cells: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_cells: array<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let array_len = arrayLength(&input_cells);
  let size = params.world_size;
  let stride = array_len / size.y;
  let coef = vec2u(stride, 1);

  let t0 = input_cells[dot(id.xy, coef)];
  let t1 = input_cells[dot((id.xy + vec2u(size.x + 0, size.y + 1)) % size, coef)];
  let t2 = input_cells[dot((id.xy + vec2u(size.x + 0, size.y - 1)) % size, coef)];
  let t3 = input_cells[dot((id.xy + vec2u(size.x - 1, size.y + 0)) % size, coef)];
  let t4 = input_cells[dot((id.xy + vec2u(size.x - 1, size.y + 1)) % size, coef)];
  let t5 = input_cells[dot((id.xy + vec2u(size.x - 1, size.y - 1)) % size, coef)];
  let t6 = input_cells[dot((id.xy + vec2u(size.x + 1, size.y + 0)) % size, coef)];
  let t7 = input_cells[dot((id.xy + vec2u(size.x + 1, size.y + 1)) % size, coef)];
  let t8 = input_cells[dot((id.xy + vec2u(size.x + 1, size.y - 1)) % size, coef)];

  let sum = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;
  let next_life = ((params.rule >> sum) >> (t0 * 8)) & 1;

  output_cells[dot(id.xy, coef)] = next_life;
}

fn calc_index(stride: u32, pos: vec2u) -> u32 {
  return pos.x + pos.y * stride;
}