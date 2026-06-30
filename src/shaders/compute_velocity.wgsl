// ============================================================================
// compute_velocity.wgsl — EvoLenia v2
// Computes the advection velocity field from mass gradients, predation,
// local soft-gravity attraction, and resource chemotaxis.
//
// Biology: Predators (high aggressivity) orient their mass flow toward
// prey (lower mass neighbors), while low-aggressivity organisms are
// weakly attracted to nearby mass concentrations (gravitational clustering).
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
@group(0) @binding(3) var<storage, read> resource_map: array<f32>;
@group(0) @binding(4) var<storage, read_write> velocity: array<vec2<f32>>;

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

    // Central differences for resource gradient ∇R (chemotaxis)
    let r_right = resource_map[idx(x + 1, y)];
    let r_left  = resource_map[idx(x - 1, y)];
    let r_up    = resource_map[idx(x, y - 1)];
    let r_down  = resource_map[idx(x, y + 1)];

    // Gradient of mass field (points toward higher mass)
    let grad_m = vec2<f32>(
        (m_right - m_left) * 0.5,
        (m_down - m_up) * 0.5
    );

    let grad_r = vec2<f32>(
        (r_right - r_left) * 0.5,
        (r_down - r_up) * 0.5
    );

    // Base velocity: mass flows along gradient, modulated by aggressivity
    // Predators (agg > 0.5) move TOWARD higher mass (prey detection)
    var vel = grad_m * agg;

    // Local soft-gravity attraction:
    // Approximate Newtonian pull from a 5x5 neighborhood with a softened
    // inverse-distance² kernel to avoid singularities/collapse.
    // This term is stronger for low-aggressivity organisms so that passive
    // lineages tend to form colonies, while predators remain more ballistic.
    var grav_force = vec2<f32>(0.0, 0.0);
    var grav_weight_sum = 0.0;
    for (var oy = -2; oy <= 2; oy = oy + 1) {
        for (var ox = -2; ox <= 2; ox = ox + 1) {
            if (ox == 0 && oy == 0) {
                continue;
            }
            let ni = idx(x + ox, y + oy);
            let m_neigh = mass[ni];
            let d = vec2<f32>(f32(ox), f32(oy));
            let d2 = dot(d, d);
            let w = 1.0 / (d2 + 0.6); // softening epsilon
            grav_force += d * (m_neigh * w);
            grav_weight_sum += w;
        }
    }
    if (grav_weight_sum > 0.0) {
        grav_force /= grav_weight_sum;
    }

    let gravity_strength = 0.18 * (1.0 - agg);
    vel += grav_force * gravity_strength;

    // Resource chemotaxis: cells drift toward nutrient-rich zones.
    // Stronger on non-predatory lineages, weaker on aggressive hunters.
    let chemo_strength = 0.22 * (1.0 - 0.5 * agg);
    vel += grad_r * chemo_strength;

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
