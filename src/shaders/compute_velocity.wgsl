// ============================================================================
// compute_velocity.wgsl — EvoLenia v2
// Computes the advection velocity field from mass gradients and predation.
//
// Biology: Predators (high aggressivity) orient their mass flow toward
// prey (lower mass neighbors), creating predator-prey spatial dynamics.
// ============================================================================

struct Params {
    width: u32,
    height: u32,
    frame: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> mass: array<f32>;
@group(0) @binding(2) var<storage, read> genome_a: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> velocity: array<vec2<f32>>;

// Toroidal indexing — wraps around edges for a borderless world
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
    let m_center = mass[i];
    let agg = genome_a[i].w; // aggressivity channel

    // Central differences for mass gradient ∇M
    let m_right = mass[idx(x + 1, y)];
    let m_left  = mass[idx(x - 1, y)];
    let m_up    = mass[idx(x, y - 1)];
    let m_down  = mass[idx(x, y + 1)];

    // Gradient of mass field (points toward higher mass)
    let grad_m = vec2<f32>(
        (m_right - m_left) * 0.5,
        (m_down - m_up) * 0.5
    );

    // Base velocity: mass flows along gradient, modulated by aggressivity
    // Predators (agg > 0.5) move TOWARD higher mass (prey detection)
    var vel = grad_m * agg;

    // Predation flux component: predators push mass toward weaker neighbors
    if (agg > 0.5 && m_center > 0.01) {
        // Check each cardinal neighbor for predation opportunity
        let neighbors = array<vec2<f32>, 4>(
            vec2<f32>(1.0, 0.0),   // right
            vec2<f32>(-1.0, 0.0),  // left
            vec2<f32>(0.0, -1.0),  // up
            vec2<f32>(0.0, 1.0)    // down
        );

        let m_neighbors = array<f32, 4>(m_right, m_left, m_up, m_down);

        var predation_vel = vec2<f32>(0.0, 0.0);
        for (var n = 0u; n < 4u; n = n + 1u) {
            let diff = m_center - m_neighbors[n];
            if (diff > 0.0) {
                // Predation: flow toward weaker neighbor
                predation_vel += neighbors[n] * agg * diff * 0.02;
            }
        }
        vel += predation_vel;
    }

    // Clamp velocity to prevent instability
    vel = clamp(vel, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0));

    velocity[i] = vel;
}
