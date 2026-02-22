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
//   5 = Metabolic Stress: Shows energy deficit — cyan=healthy, magenta=starving
//   6 = Advection Flux: Velocity field magnitude — blue=still, yellow=fast
//   7 = Trophic Roles: Prey(green) / Opportunist(blue) / Predator(red)
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
@group(0) @binding(5) var<storage, read> velocity: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> resource_map: array<f32>;

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

    // Mode 5: Metabolic Stress — energy deficit visualization
    // Cyan = healthy (high energy), Magenta = starving, overlaid on resource landscape
    if render_params.visualization_mode == 5u {
        let r_val = resource_map[idx];
        let resource_bg = vec3<f32>(0.02, 0.08 * r_val, 0.02); // dim green for resource base
        if (m > 0.01) {
            let stress = 1.0 - clamp(e / 0.3, 0.0, 1.0); // 0=healthy, 1=starving
            let healthy_col = vec3<f32>(0.0, 0.9, 0.9);   // cyan
            let starving_col = vec3<f32>(0.9, 0.0, 0.7);  // magenta
            let stress_col = mix(healthy_col, starving_col, stress);
            let color = mix(resource_bg, stress_col, m);
            return vec4<f32>(color, 1.0);
        }
        return vec4<f32>(resource_bg, 1.0);
    }

    // Mode 6: Advection Flux — velocity field magnitude
    // Blue = stationary, Yellow = high flux, with directional tint
    if render_params.visualization_mode == 6u {
        let vel = velocity[idx];
        let speed = length(vel);
        let norm_speed = clamp(speed * 20.0, 0.0, 1.0); // scale for visibility
        // Direction-dependent color: hue from atan2
        let angle = atan2(vel.y, vel.x); // -π to π
        let hue = (angle / 6.2832 + 0.5); // 0 to 1
        let flux_col = hsv2rgb(hue, 0.8, norm_speed);
        let still_col = vec3<f32>(0.05, 0.05, 0.15);
        let color = mix(still_col, flux_col, clamp(norm_speed + m * 0.3, 0.0, 1.0));
        return vec4<f32>(color, 1.0);
    }

    // Mode 7: Trophic Roles — multi-level trophic classification
    // Green = passive prey (agg<0.2), Blue = opportunist (0.2-0.5), Red = predator (>0.5)
    // Brightness = mass, saturation = specialization (low sigma = specialist)
    if render_params.visualization_mode == 7u {
        if (m > 0.01) {
            let agg_v = ga.w;
            let specialization = clamp(1.0 - ga.z / 0.2, 0.0, 1.0);
            var role_col: vec3<f32>;
            if (agg_v < 0.2) {
                // Prey: green → lime, specialist prey are more saturated
                role_col = vec3<f32>(0.1, 0.85, 0.15);
            } else if (agg_v < 0.5) {
                // Opportunist: blue-teal, interpolated
                let t = (agg_v - 0.2) / 0.3;
                role_col = mix(vec3<f32>(0.1, 0.7, 0.6), vec3<f32>(0.3, 0.3, 0.9), t);
            } else {
                // Predator: orange → red
                let t = (agg_v - 0.5) / 0.5;
                role_col = mix(vec3<f32>(1.0, 0.5, 0.0), vec3<f32>(1.0, 0.0, 0.0), t);
            }
            let sat = mix(0.5, 1.0, specialization);
            let final_col = mix(vec3<f32>(0.5), role_col, sat);
            let color = mix(bg, final_col, m);
            return vec4<f32>(color, 1.0);
        }
        return vec4<f32>(bg, 1.0);
    }

    // Fallback (should never reach)
    return vec4<f32>(bg, 1.0);
}
