// ============================================================================
// compute_resources.wgsl — EvoLenia v2
// Gray-Scott inspired resource dynamics: diffusion + regeneration - consumption
//
// Biology: Nutrients diffuse spatially, regenerate slowly (feed_rate),
// and are consumed by organisms. This creates spatial selection pressure:
// areas depleted by organisms become deserts, pushing evolution to
// disperse or become more efficient.
// ============================================================================

struct Params {
    width: u32,
    height: u32,
    frame: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> mass: array<f32>;
@group(0) @binding(2) var<storage, read_write> resource_map: array<f32>;

// Toroidal indexing
fn idx(x: i32, y: i32) -> u32 {
    let wx = ((x % i32(params.width)) + i32(params.width)) % i32(params.width);
    let wy = ((y % i32(params.height)) + i32(params.height)) % i32(params.height);
    return u32(wy) * params.width + u32(wx);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let i = idx(x, y);
    let r = resource_map[i];
    let m = mass[i];

    // Discrete Laplacian for diffusion (5-point stencil)
    let r_right = resource_map[idx(x + 1, y)];
    let r_left  = resource_map[idx(x - 1, y)];
    let r_up    = resource_map[idx(x, y - 1)];
    let r_down  = resource_map[idx(x, y + 1)];

    let laplacian = (r_right + r_left + r_up + r_down - 4.0 * r) / 4.0;

    // Gray-Scott dynamics:
    // - Diffusion: nutrients spread spatially (D_R = 0.08)
    // - Feed: nutrients regenerate toward 1.0 (feed_rate = 0.010) — slower regeneration
    // - Consumption: organisms consume nutrients proportional to their mass
    let diffusion     = 0.08 * laplacian;
    let feed          = 0.010 * (1.0 - r);
    let consumed      = r * m * 0.08;

    let r_new = clamp(r + diffusion + feed - consumed, 0.0, 1.0);

    resource_map[i] = r_new;
}
