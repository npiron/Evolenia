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
//   8 = Debug Raw: Direct buffer values — R=mass, G=energy, B=resources
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
    show_legend: u32,
    time: f32,       // seconds, for animated effects
    _pad: f32,       // alignment padding
}

struct CameraUniforms {
    offset: vec2<f32>,      // world-UV pan offset
    zoom: f32,              // zoom factor (>1 = zoomed in)
    aspect_ratio: f32,      // window aspect ratio
    world_aspect: f32,      // world aspect ratio
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
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
    
    let m_v = v - c;
    return rgb + vec3<f32>(m_v);
}

// --- Visual quality utilities ---

/// ACES filmic tone mapping — richer, less harsh colours
fn aces_tone_map(col: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((col * (a * col + b)) / (col * (c * col + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Subtle vignette: darkens screen edges for a cinematic look
fn vignette(uv: vec2<f32>, col: vec3<f32>) -> vec3<f32> {
    let d = length(uv - vec2<f32>(0.5)) * 1.2;
    let vig = 1.0 - d * d * 0.45;
    return col * clamp(vig, 0.0, 1.0);
}

/// Toroidal neighbour index helpers (uses module-scope render_params uniform)
fn idx_right(x: u32, y: u32) -> u32 { return y * render_params.width + (x + 1u) % render_params.width; }
fn idx_left(x: u32, y: u32)  -> u32 { return y * render_params.width + (x + render_params.width - 1u) % render_params.width; }
fn idx_down(x: u32, y: u32)  -> u32 { return ((y + 1u) % render_params.height) * render_params.width + x; }
fn idx_up(x: u32, y: u32)    -> u32 { return ((y + render_params.height - 1u) % render_params.height) * render_params.width + x; }

/// Apply tone mapping + vignette to a final colour
fn finalize_color(col: vec3<f32>, uv: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(vignette(uv, aces_tone_map(col)), 1.0);
}

// --- Shared blend helper ---
/// Blend organism colour over background with smoothstep for organic edges.
fn blend_over_background(bg: vec3<f32>, color: vec3<f32>, m: f32) -> vec3<f32> {
    let alpha = smoothstep(0.0, 0.12, m); // soft transition instead of linear
    return mix(bg, color, alpha);
}

// --- Legend helpers (screen-space) ---

/// Horizontal color bar at given normalized y position.
fn legend_bar(px: f32, py: f32, bar_y: f32, bar_h: f32, left_col: vec3<f32>, right_col: vec3<f32>) -> vec4<f32> {
    if (py >= bar_y && py <= bar_y + bar_h) {
        return vec4<f32>(mix(left_col, right_col, px), 1.0);
    }
    return vec4<f32>(0.0);
}

/// Small square color swatch.
fn legend_swatch(px: f32, py: f32, sx: f32, sy: f32, ss: f32, col: vec3<f32>) -> vec4<f32> {
    if (px >= sx && px <= sx + ss && py >= sy && py <= sy + ss) {
        return vec4<f32>(col, 1.0);
    }
    return vec4<f32>(0.0);
}

/// Hue bar (uses HSV conversion directly on px).
fn legend_hue_bar(px: f32, py: f32, bar_y: f32, bar_h: f32) -> vec4<f32> {
    if (py >= bar_y && py <= bar_y + bar_h) {
        return vec4<f32>(hsv2rgb(px, 0.8, 0.9), 1.0);
    }
    return vec4<f32>(0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Correct aspect ratio: scale UV so world appears square regardless of window shape
    let centered = in.uv - vec2<f32>(0.5, 0.5);
    
    // Aspect ratio correction: if window is wider than world, add horizontal letterbox
    var corrected = centered;
    let ratio_correction = camera.aspect_ratio / camera.world_aspect;
    if (ratio_correction > 1.0) {
        // Window is wider than world - scale X to fit
        corrected.x = corrected.x * ratio_correction;
    } else {
        // Window is taller than world - scale Y to fit
        corrected.y = corrected.y / ratio_correction;
    }
    
    let world_uv = corrected / camera.zoom + vec2<f32>(0.5, 0.5) + camera.offset;

    // Outside the [0,1] world bounds: static dark background
    let outside_bg = vec3<f32>(0.06, 0.06, 0.09);
    if (world_uv.x < 0.0 || world_uv.x > 1.0 || world_uv.y < 0.0 || world_uv.y > 1.0) {
        return vec4<f32>(vignette(in.uv, aces_tone_map(outside_bg)), 1.0);
    }

    // Clamp to world bounds (no toroidal wrap for rendering)
    let wx = world_uv.x;
    let wy = world_uv.y;

    let px = u32(wx * f32(render_params.width));
    let py = u32(wy * f32(render_params.height));

    let cx = min(px, render_params.width - 1u);
    let cy = min(py, render_params.height - 1u);

    let idx = cy * render_params.width + cx;
    let m = mass[idx];
    let e = energy[idx];
    let ga = genome_a[idx]; // r, mu, sigma, aggressivity

    // Static dark background — scientifically clean, no distractions
    let bg = vec3<f32>(0.015, 0.015, 0.04);

    // --- Local glow: empty pixels near organisms get a soft halo ---
    var glow = vec3<f32>(0.0);
    if (m < 0.02) {
        let n_r = mass[idx_right(cx, cy)];
        let n_l = mass[idx_left(cx, cy)];
        let n_d = mass[idx_down(cx, cy)];
        let n_u = mass[idx_up(cx, cy)];
        let glow_mass = (n_r + n_l + n_d + n_u) * 0.25;
        if (glow_mass > 0.01) {
            let ga_r = genome_a[idx_right(cx, cy)];
            let ga_l = genome_a[idx_left(cx, cy)];
            let ga_d = genome_a[idx_down(cx, cy)];
            let ga_u = genome_a[idx_up(cx, cy)];
            let glow_ga = (ga_r + ga_l + ga_d + ga_u) * 0.25;
            let glow_color = vec3<f32>(
                clamp(glow_ga.x / 15.0, 0.0, 1.0),
                clamp(glow_ga.y * 5.0, 0.0, 1.0),
                clamp(glow_ga.z / 0.06, 0.0, 1.0)
            );
            glow = glow_color * glow_mass * 0.18;
        }
    }

    // Mode 0: Species Color
    if render_params.visualization_mode == 0u {
        let species_color = vec3<f32>(
            clamp(ga.x / 15.0, 0.0, 1.0),   // R = perception radius (max 15)
            clamp(ga.y * 5.0, 0.0, 1.0),     // G = growth center μ (scaled: 0.15 → 0.75)
            clamp(ga.z / 0.06, 0.0, 1.0)     // B = growth width σ (scaled for Lenia range)
        );
        let predator_glow = step(0.7, ga.w) * vec3<f32>(1.0, 0.5, 0.0);
        let final_color = clamp(species_color + predator_glow * 0.3, vec3<f32>(0.0), vec3<f32>(1.0));
        return finalize_color(blend_over_background(bg, final_color, m) + glow, in.uv);
    }
    
    // Mode 1: Energy Heatmap (blue = low, red = high)
    if render_params.visualization_mode == 1u {
        let heat_color = vec3<f32>(e, 0.2, 1.0 - e); // Blue -> Purple -> Red
        return finalize_color(blend_over_background(bg, heat_color, m) + glow, in.uv);
    }
    
    // Mode 2: Mass Density (grayscale)
    if render_params.visualization_mode == 2u {
        let gray = vec3<f32>(m);
        return finalize_color(gray + glow, in.uv);
    }
    
    // Mode 3: Genetic Diversity (hue from genome hash)
    if render_params.visualization_mode == 3u {
        // Hash genome to a hue (0-1)
        let genome_hash = fract((ga.x * 0.1 + ga.y * 0.3 + ga.z * 3.0 + ga.w * 0.7) * 43758.5453);
        let diversity_color = hsv2rgb(genome_hash, 0.8, 0.9);
        return finalize_color(blend_over_background(bg, diversity_color, m) + glow, in.uv);
    }
    
    // Mode 4: Predator/Prey (red = predator, green = prey)
    if render_params.visualization_mode == 4u {
        let predator_color = vec3<f32>(1.0, 0.0, 0.0); // Red
        let prey_color = vec3<f32>(0.0, 1.0, 0.0);     // Green
        let species_color = mix(prey_color, predator_color, ga.w);
        return finalize_color(blend_over_background(bg, species_color, m) + glow, in.uv);
    }

    // Mode 5: Metabolic Stress — energy deficit visualization
    if render_params.visualization_mode == 5u {
        let r_val = resource_map[idx];
        let resource_bg = vec3<f32>(0.02, 0.08 * r_val, 0.02); // dim green for resource base
        if (m > 0.01) {
            let stress = 1.0 - clamp(e / 0.3, 0.0, 1.0); // 0=healthy, 1=starving
            let healthy_col = vec3<f32>(0.0, 0.9, 0.9);   // cyan
            let starving_col = vec3<f32>(0.9, 0.0, 0.7);  // magenta
            let stress_col = mix(healthy_col, starving_col, stress);
            return finalize_color(blend_over_background(resource_bg, stress_col, m) + glow, in.uv);
        }
        return finalize_color(resource_bg + glow, in.uv);
    }

    // Mode 6: Advection Flux — velocity field magnitude
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
        return finalize_color(color + glow, in.uv);
    }

    // Mode 7: Trophic Roles — multi-level trophic classification
    if render_params.visualization_mode == 7u {
        if (m > 0.01) {
            let agg_v = ga.w;
            let specialization = clamp(1.0 - ga.z / 0.2, 0.0, 1.0);
            var role_col: vec3<f32>;
            if (agg_v < 0.2) {
                role_col = vec3<f32>(0.1, 0.85, 0.15);
            } else if (agg_v < 0.5) {
                let t = (agg_v - 0.2) / 0.3;
                role_col = mix(vec3<f32>(0.1, 0.7, 0.6), vec3<f32>(0.3, 0.3, 0.9), t);
            } else {
                let t = (agg_v - 0.5) / 0.5;
                role_col = mix(vec3<f32>(1.0, 0.5, 0.0), vec3<f32>(1.0, 0.0, 0.0), t);
            }
            let sat = mix(0.5, 1.0, specialization);
            let final_col = mix(vec3<f32>(0.5), role_col, sat);
            return finalize_color(blend_over_background(bg, final_col, m) + glow, in.uv);
        }
        return finalize_color(bg + glow, in.uv);
    }

    // Mode 8: Debug Raw — direct rendering of raw buffer values (no aesthetic filtering)
    // Useful for shader debugging: red = mass, green = energy, blue = resources
    if render_params.visualization_mode == 8u {
        let r_val = resource_map[idx];
        let raw_col = vec3<f32>(m, e, r_val);
        return finalize_color(raw_col + glow, in.uv);
    }

    // --- Legend overlay (bottom-left screen-space) ---
    if (render_params.show_legend == 1u) {
        let lx = in.uv.x;       // 0..1, left→right
        let ly = 1.0 - in.uv.y; // 0..1, bottom→top (uv.y is top-down)

        // Legend panel bounds — centered bottom, avoids egui side panels
        let leg_left = 0.32;
        let leg_bot = 0.006;
        let leg_w = 0.22;
        let leg_h = 0.14;

        if (lx >= leg_left && lx <= leg_left + leg_w &&
            ly >= leg_bot && ly <= leg_bot + leg_h) {

            let bg_col = vec3<f32>(0.12, 0.12, 0.18);
            let px = (lx - leg_left) / leg_w;
            let py = (ly - leg_bot) / leg_h;

            // Draw thin border around legend panel
            let border = f32(px < 0.006 || px > 0.994 || py < 0.006 || py > 0.994);
            if (border > 0.5) {
                return vec4<f32>(0.55, 0.55, 0.60, 1.0);
            }

            var result = vec4<f32>(0.0);
            let mode = render_params.visualization_mode;

            if (mode == 0u) {
                let sw_r = legend_swatch(px, py, 0.02, 0.62, 0.055, vec3<f32>(1.0, 0.0, 0.0));
                let sw_g = legend_swatch(px, py, 0.10, 0.62, 0.055, vec3<f32>(0.0, 1.0, 0.0));
                let sw_b = legend_swatch(px, py, 0.18, 0.62, 0.055, vec3<f32>(0.05, 0.05, 1.0));
                result = vec4<f32>(sw_r.rgb + sw_g.rgb + sw_b.rgb, 1.0);
            }
            else if (mode == 1u) {
                result = legend_bar(px, py, 0.50, 0.18, vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0));
            }
            else if (mode == 2u) {
                result = legend_bar(px, py, 0.50, 0.18, vec3<f32>(0.05), vec3<f32>(1.0));
            }
            else if (mode == 3u) {
                result = legend_hue_bar(px, py, 0.50, 0.18);
            }
            else if (mode == 4u) {
                result = legend_bar(px, py, 0.50, 0.18, vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0));
            }
            else if (mode == 5u) {
                result = legend_bar(px, py, 0.50, 0.18, vec3<f32>(0.0, 0.9, 0.9), vec3<f32>(0.9, 0.0, 0.7));
            }
            else if (mode == 6u) {
                // Direction wheel
                let cx = 0.50; let cy = 0.55; let rad = 0.14;
                let dx = px - cx; let dy = py - cy;
                let dist = sqrt(dx * dx + dy * dy);
                if (dist < rad && dist > rad * 0.65) {
                    let ang = atan2(dy, dx);
                    result = vec4<f32>(hsv2rgb((ang / 6.2832 + 0.5), 0.8, 0.7), 1.0);
                }
            }
            else if (mode == 7u) {
                let sw1 = legend_swatch(px, py, 0.01, 0.58, 0.065, vec3<f32>(0.1, 0.85, 0.15));
                let sw2 = legend_swatch(px, py, 0.10, 0.58, 0.065, vec3<f32>(0.3, 0.3, 0.9));
                let sw3 = legend_swatch(px, py, 0.19, 0.58, 0.065, vec3<f32>(1.0, 0.0, 0.0));
                result = vec4<f32>(sw1.rgb + sw2.rgb + sw3.rgb, 1.0);
            }
            else if (mode == 8u) {
                let sw_r = legend_swatch(px, py, 0.01, 0.58, 0.065, vec3<f32>(1.0, 0.0, 0.0));
                let sw_g = legend_swatch(px, py, 0.10, 0.58, 0.065, vec3<f32>(0.0, 1.0, 0.0));
                let sw_b = legend_swatch(px, py, 0.19, 0.58, 0.065, vec3<f32>(0.0, 0.3, 1.0));
                result = vec4<f32>(sw_r.rgb + sw_g.rgb + sw_b.rgb, 1.0);
            }

            if (result.a > 0.0) {
                return vec4<f32>(mix(bg_col, result.rgb, 0.9), 1.0);
            }
            return vec4<f32>(bg_col, 1.0);
        }
    }

    // Fallback (should never reach)
    return finalize_color(bg + glow, in.uv);
}
