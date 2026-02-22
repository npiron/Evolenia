// ============================================================================
// render.wgsl — EvoLenia v2
// Visualization: Multiple rendering modes for scientific observation.
//
// Modes:
//   0 = Species Color: RGB = genome(r, μ, σ), orange glow = predators
//   1 = Energy Heatmap: Blue = low energy, Red = high energy
//   2 = Mass Density: Grayscale intensity
//   3 = Genetic Diversity: Color variation by local genome variance
//   4 = Predator/Prey: Red = high aggressivity, Green = passive
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
    visualization_mode: u32,
    _pad: u32,
}

struct CameraUniforms {
    offset: vec2<f32>, // world-UV pan offset
    zoom: f32,         // zoom factor (>1 = zoomed in)
    _pad: f32,
}

@group(0) @binding(0) var<uniform> render_params: RenderParams;
@group(0) @binding(1) var<storage, read> mass: array<f32>;
@group(0) @binding(2) var<storage, read> energy: array<f32>;
@group(0) @binding(3) var<storage, read> genome_a: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> camera: CameraUniforms;

// HSV to RGB conversion for diversity visualization
fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - abs((h6 % 2.0) - 1.0));
    
    var rgb = vec3<f32>(0.0);
    if h6 < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if h6 < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if h6 < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if h6 < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if h6 < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    let m = v - c;
    return rgb + vec3<f32>(m);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply camera: center UV, scale by 1/zoom, add pan offset, re-center
    let centered = in.uv - vec2<f32>(0.5, 0.5);
    let world_uv = centered / camera.zoom + vec2<f32>(0.5, 0.5) + camera.offset;

    // Toroidal wrap so the world tiles seamlessly when panning
    let wx = ((world_uv.x % 1.0) + 1.0) % 1.0;
    let wy = ((world_uv.y % 1.0) + 1.0) % 1.0;

    let px = u32(wx * f32(render_params.width));
    let py = u32(wy * f32(render_params.height));

    let cx = min(px, render_params.width - 1u);
    let cy = min(py, render_params.height - 1u);

    let idx = cy * render_params.width + cx;
    let m = mass[idx];
    let e = energy[idx];
    let ga = genome_a[idx]; // r, mu, sigma, aggressivity

    let bg = vec3<f32>(0.02, 0.02, 0.05); // Dark background

    // Mode 0: Species Color
    if render_params.visualization_mode == 0u {
        let species_color = vec3<f32>(
            clamp(ga.x / 16.0, 0.0, 1.0),   // R = perception radius
            clamp(ga.y, 0.0, 1.0),           // G = growth center μ
            clamp(ga.z / 0.3, 0.0, 1.0)      // B = growth width σ
        );
        let predator_glow = step(0.7, ga.w) * vec3<f32>(1.0, 0.5, 0.0);
        let final_color = clamp(species_color + predator_glow * 0.3, vec3<f32>(0.0), vec3<f32>(1.0));
        let color = mix(bg, final_color, m);
        return vec4<f32>(color, 1.0);
    }
    
    // Mode 1: Energy Heatmap (blue = low, red = high)
    if render_params.visualization_mode == 1u {
        let heat_color = vec3<f32>(e, 0.2, 1.0 - e); // Blue -> Purple -> Red
        let color = mix(bg, heat_color, m);
        return vec4<f32>(color, 1.0);
    }
    
    // Mode 2: Mass Density (grayscale)
    if render_params.visualization_mode == 2u {
        let gray = vec3<f32>(m);
        return vec4<f32>(gray, 1.0);
    }
    
    // Mode 3: Genetic Diversity (hue from genome hash)
    if render_params.visualization_mode == 3u {
        // Hash genome to a hue (0-1)
        let genome_hash = fract((ga.x * 0.1 + ga.y * 0.3 + ga.z * 3.0 + ga.w * 0.7) * 43758.5453);
        let diversity_color = hsv2rgb(genome_hash, 0.8, 0.9);
        let color = mix(bg, diversity_color, m);
        return vec4<f32>(color, 1.0);
    }
    
    // Mode 4: Predator/Prey (red = predator, green = prey)
    if render_params.visualization_mode == 4u {
        let predator_color = vec3<f32>(1.0, 0.0, 0.0); // Red
        let prey_color = vec3<f32>(0.0, 1.0, 0.0);     // Green
        let species_color = mix(prey_color, predator_color, ga.w);
        let color = mix(bg, species_color, m);
        return vec4<f32>(color, 1.0);
    }

    // Fallback (should never reach)
    return vec4<f32>(bg, 1.0);
}
