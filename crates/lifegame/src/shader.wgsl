struct Params {
  world_size: vec2u,
  // https://conwaylife.com/wiki/Rule_integer
  rule: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

struct VertexInput {
    @location(0) cell: u32,
    @location(1) position: vec2f,
    @builtin(instance_index) index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f, 
    @location(0) cell: u32,
};

@vertex
fn main_vs(
    model: VertexInput,
) -> VertexOutput {
    let cell_size = vec2f(2.0, 2.0) / vec2f(params.world_size);
    let cell_pos = vec2u(model.index % params.world_size.x, model.index / params.world_size.x);
    let out_pos = vec2f(-1, -1) + cell_size * vec2f(cell_pos) + model.position * cell_size;

    var out: VertexOutput;
    out.clip_position = vec4f(out_pos, 0, 1);
    out.cell = model.cell;
    return out;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4f {
  let c = f32(in.cell);

  return vec4f(c, c, c, 1);
}