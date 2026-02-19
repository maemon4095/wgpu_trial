struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn main_vs(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.uv = model.uv;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader

@group(0) @binding(0)
var tex: texture_2d<f32>;
@group(0) @binding(1)
var s: sampler;

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(tex, s, in.uv);
}