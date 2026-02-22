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
        var predation_vel = vec2<f32>(0.0, 0.0);

        // right
        let diff0 = m_center - m_right;
        if (diff0 > 0.0) { predation_vel += vec2<f32>(1.0, 0.0) * agg * diff0 * 0.008; }
        // left
        let diff1 = m_center - m_left;
        if (diff1 > 0.0) { predation_vel += vec2<f32>(-1.0, 0.0) * agg * diff1 * 0.008; }
        // up
        let diff2 = m_center - m_up;
        if (diff2 > 0.0) { predation_vel += vec2<f32>(0.0, -1.0) * agg * diff2 * 0.008; }
        // down
        let diff3 = m_center - m_down;
        if (diff3 > 0.0) { predation_vel += vec2<f32>(0.0, 1.0) * agg * diff3 * 0.008; }

        vel += predation_vel;
    }

    // Clamp velocity to prevent instability
    vel = clamp(vel, vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0));

    velocity[i] = vel;
}
