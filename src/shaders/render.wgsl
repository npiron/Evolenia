// ============================================================================
// render.wgsl — EvoLenia v2
// Visualization: maps genome channels to colors for scientific observation.
//
// Color mapping:
//   R = perception radius (r/16)   → red = small creatures
//   G = growth center (μ)          → green = ecological niche
//   B = growth width (σ)           → blue = generalist/specialist
//   Alpha = mass density
//
// Predator highlighting: organisms with aggressivity > 0.7 get an orange glow,
// making predator-prey dynamics visually apparent.
// ============================================================================

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen quad: 2 triangles, no vertex buffer needed
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Map vertex_index 0..5 to two triangles covering [-1,1]²
    var pos: vec2<f32>;
    var uv: vec2<f32>;

    switch vertex_index {
        case 0u: { pos = vec2<f32>(-1.0, -1.0); uv = vec2<f32>(0.0, 1.0); }
        case 1u: { pos = vec2<f32>( 1.0, -1.0); uv = vec2<f32>(1.0, 1.0); }
        case 2u: { pos = vec2<f32>(-1.0,  1.0); uv = vec2<f32>(0.0, 0.0); }
        case 3u: { pos = vec2<f32>(-1.0,  1.0); uv = vec2<f32>(0.0, 0.0); }
        case 4u: { pos = vec2<f32>( 1.0, -1.0); uv = vec2<f32>(1.0, 1.0); }
        case 5u: { pos = vec2<f32>( 1.0,  1.0); uv = vec2<f32>(1.0, 0.0); }
        default: { pos = vec2<f32>(0.0, 0.0); uv = vec2<f32>(0.0, 0.0); }
    }

    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uv;
    return out;
}

struct RenderParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> render_params: RenderParams;
@group(0) @binding(1) var<storage, read> mass: array<f32>;
@group(0) @binding(2) var<storage, read> genome_a: array<vec4<f32>>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = u32(in.uv.x * f32(render_params.width));
    let py = u32(in.uv.y * f32(render_params.height));

    // Clamp to valid pixel range
    let cx = min(px, render_params.width - 1u);
    let cy = min(py, render_params.height - 1u);

    let idx = cy * render_params.width + cx;
    let m = mass[idx];
    let ga = genome_a[idx];

    // Species color from genome (clamped to [0,1] to avoid oversaturation)
    let species_color = vec3<f32>(
        clamp(ga.x / 16.0, 0.0, 1.0),   // R = perception radius (normalized)
        clamp(ga.y, 0.0, 1.0),           // G = growth center μ (ecological niche)
        clamp(ga.z / 0.3, 0.0, 1.0)      // B = growth width σ (normalized to [0,1])
    );

    // Predator glow: high aggressivity (> 0.7) → orange highlight
    let predator_glow = step(0.7, ga.w) * vec3<f32>(1.0, 0.5, 0.0);
    let final_color = clamp(species_color + predator_glow * 0.3, vec3<f32>(0.0), vec3<f32>(1.0));

    // Background: dark when no mass, colored when alive
    let bg = vec3<f32>(0.02, 0.02, 0.05); // Very dark blue background
    let color = mix(bg, final_color, m);

    return vec4<f32>(color, 1.0);
}
