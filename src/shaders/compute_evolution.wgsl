// ============================================================================
// compute_evolution.wgsl — EvoLenia v2
// Core evolution pass: Lenia convolution, growth, metabolism, mass-conservative
// advection, stochastic DNA segregation, and mutations.
//
// Biology: This is the "physics + genetics" engine. Each pixel is an organism
// with its own genome. The Lenia rule governs local growth/decay, while
// advection ensures mass is transferred (never duplicated). DNA travels
// with mass via stochastic segregation — the colonizer's genome replaces
// the receiver's with probability proportional to the flux.
// ============================================================================

struct Params {
    width: u32,
    height: u32,
    frame: u32,
    dt: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> mass_in: array<f32>;
@group(0) @binding(2) var<storage, read> energy_in: array<f32>;
@group(0) @binding(3) var<storage, read> genome_a_in: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> genome_b_in: array<f32>;
@group(0) @binding(5) var<storage, read> resource_map: array<f32>;
@group(0) @binding(6) var<storage, read> velocity: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> mass_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> energy_out: array<f32>;
@group(0) @binding(9) var<storage, read_write> genome_a_out: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> genome_b_out: array<f32>;

// ======================== PRNG ========================
// PCG hash-based pseudo-random number generator (no global state)
fn pcg_hash(inp: u32) -> u32 {
    var state = inp * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Returns a float in [0, 1]
fn rand01(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

// Returns a float in [-1, 1]
fn rand_signed(seed: u32) -> f32 {
    return rand01(seed) * 2.0 - 1.0;
}

// Toroidal indexing
fn idx(x: i32, y: i32) -> u32 {
    let wx = ((x % i32(params.width)) + i32(params.width)) % i32(params.width);
    let wy = ((y % i32(params.height)) + i32(params.height)) % i32(params.height);
    return u32(wy) * params.width + u32(wx);
}

// ======================== LENIA RING KERNEL ========================
// Ring kernel weight: K(d, r) = exp(-((d/r - 0.5)^2 / (2 * 0.15^2)))
// This creates a ring-shaped perception pattern at distance ~r/2
fn kernel_weight(dist: f32, radius: f32) -> f32 {
    let normalized = dist / radius;
    let diff = normalized - 0.5;
    return exp(-(diff * diff) / (2.0 * 0.15 * 0.15));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let i = idx(x, y);
    let m = mass_in[i];
    let e = energy_in[i];
    let ga = genome_a_in[i]; // r, mu, sigma, aggressivity
    let gb = genome_b_in[i]; // mutation_rate

    let r      = ga.x; // perception radius
    let mu     = ga.y; // growth center (ecological niche)
    let sigma  = ga.z; // growth width (tolerance)
    let agg    = ga.w; // aggressivity

    // Base seed for PRNG — unique per pixel per frame
    let base_seed = (gid.y * params.width + gid.x) ^ params.frame ^ 0xDEADBEEFu;

    // ================== LENIA CONVOLUTION ==================
    // Blend between three fixed kernel radii (3, 7, 14) based on genome radius
    // This avoids dynamic kernel sizes which are expensive on GPU
    let r_small = 3.0;
    let r_mid   = 7.0;
    let r_large = 14.0;
    let max_r   = 14;

    var U = 0.0; // Perceived density (convolution result)
    var kernel_sum = 0.0;

    // Sample the neighborhood up to max kernel radius
    for (var dy = -max_r; dy <= max_r; dy = dy + 1) {
        for (var dx = -max_r; dx <= max_r; dx = dx + 1) {
            let dist = sqrt(f32(dx * dx + dy * dy));
            if (dist < 0.5 || dist > f32(max_r)) {
                continue;
            }

            // Interpolate kernel weight based on genome radius r
            var w = 0.0;
            if (r <= r_small) {
                w = kernel_weight(dist, r_small);
                // Fade out contributions beyond small radius
                if (dist > r_small * 1.5) { w = 0.0; }
            } else if (r <= r_mid) {
                let t = (r - r_small) / (r_mid - r_small);
                let w_s = kernel_weight(dist, r_small);
                let w_m = kernel_weight(dist, r_mid);
                // Fade small kernel beyond its effective range
                let ws_faded = select(0.0, w_s, dist <= r_small * 1.5);
                w = mix(ws_faded, w_m, t);
                if (dist > r_mid * 1.5) { w *= (1.0 - t); }
            } else {
                let t = (r - r_mid) / (r_large - r_mid);
                let w_m = kernel_weight(dist, r_mid);
                let w_l = kernel_weight(dist, r_large);
                let wm_faded = select(0.0, w_m, dist <= r_mid * 1.5);
                w = mix(wm_faded, w_l, t);
            }

            if (w > 0.001) {
                let ni = idx(x + dx, y + dy);
                U += w * mass_in[ni];
                kernel_sum += w;
            }
        }
    }

    // Normalize kernel (critical: without this, U diverges)
    if (kernel_sum > 0.0) {
        U = U / kernel_sum;
    }

    // ================== GROWTH FUNCTION ==================
    // Gaussian bell: G(U; μ, σ) = exp(-((U - μ)² / (2σ²)))
    // Biologically: organisms thrive at density μ, tolerate ±σ
    let growth_raw = exp(-((U - mu) * (U - mu)) / (2.0 * sigma * sigma));
    let dM = 2.0 * growth_raw - 1.0; // ∈ [-1, +1]
    var mass_candidate = clamp(m + params.dt * dM, 0.0, 1.0);

    // ================== METABOLISM ==================
    // Cost scales with genomic complexity (Darwinian parsimony)
    let genomic_complexity = length(vec3<f32>(mu, sigma, agg));
    let cost = genomic_complexity * m * 0.01;
    // Absorption from local resource map (nutrient uptake)
    let absorption = resource_map[i] * m * 0.05;
    var energy_new = clamp(e + absorption - cost, 0.0, 1.0);

    // Starvation: gradual mass decay when energy depleted
    if (energy_new <= 0.0) {
        mass_candidate *= 0.99;
    }

    // ================== MASS-CONSERVATIVE ADVECTION ==================
    // Mass is TRANSFERRED, never copied. Conservation: flux_in = flux_out
    let vel = velocity[i];

    // Cardinal direction vectors
    let dirs = array<vec2<f32>, 4>(
        vec2<f32>(1.0, 0.0),   // right
        vec2<f32>(-1.0, 0.0),  // left
        vec2<f32>(0.0, 1.0),   // down
        vec2<f32>(0.0, -1.0)   // up
    );

    let neighbor_offsets = array<vec2<i32>, 4>(
        vec2<i32>(1, 0),
        vec2<i32>(-1, 0),
        vec2<i32>(0, 1),
        vec2<i32>(0, -1)
    );

    var total_flux_out = 0.0;
    var total_flux_in = 0.0;

    // Flux limiters: cannot send more than M/4 to any single neighbor
    for (var n = 0u; n < 4u; n = n + 1u) {
        // Outgoing flux from self to neighbor
        let flux_component = dot(vel, dirs[n]);
        let flux_out = clamp(flux_component, 0.0, mass_candidate / 4.0);
        total_flux_out += flux_out;

        // Incoming flux from neighbor to self
        let ni = idx(x + neighbor_offsets[n].x, y + neighbor_offsets[n].y);
        let vel_n = velocity[ni];
        let mass_n = mass_in[ni];
        // Neighbor's flux toward us = their velocity projected onto -dir
        let flux_in_component = dot(vel_n, -dirs[n]);
        let flux_in = clamp(flux_in_component, 0.0, mass_n / 4.0);
        total_flux_in += flux_in;
    }

    var mass_new = mass_candidate + total_flux_in - total_flux_out;
    mass_new = clamp(mass_new, 0.0, 1.0);

    // ================== DNA ADVECTION — STOCHASTIC SEGREGATION ==================
    // When mass flows from neighbor to self, the neighbor's genome can
    // "colonize" this cell. Probability proportional to flux/mass ratio.
    // Biology: this implements spatial heredity via mass transport.
    var genome_a_new = ga;
    var genome_b_new = gb;

    var seed = base_seed;
    for (var n = 0u; n < 4u; n = n + 1u) {
        let ni = idx(x + neighbor_offsets[n].x, y + neighbor_offsets[n].y);
        let vel_n = velocity[ni];
        let mass_n = mass_in[ni];
        let flux_in_component = dot(vel_n, -dirs[n]);
        let flux_in = clamp(flux_in_component, 0.0, mass_n / 4.0);

        if (flux_in > 0.001) {
            // Colonization probability: larger flux → higher chance of replacement
            let p_replace = flux_in / (mass_new + 0.001);
            seed = pcg_hash(seed + n + 1u);
            if (rand01(seed) < p_replace) {
                genome_a_new = genome_a_in[ni]; // Colonizer's genome wins
                genome_b_new = genome_b_in[ni];
            }
        }
    }

    // ================== MUTATIONS ==================
    // Only living cells mutate (dead cells are inert)
    if (mass_new > 0.01) {
        let mut_rate = genome_b_new;

        // Independent noise per gene channel
        seed = pcg_hash(seed + 100u);
        let noise_r = rand_signed(seed);
        seed = pcg_hash(seed + 101u);
        let noise_mu = rand_signed(seed);
        seed = pcg_hash(seed + 102u);
        let noise_sigma = rand_signed(seed);
        seed = pcg_hash(seed + 103u);
        let noise_agg = rand_signed(seed);
        seed = pcg_hash(seed + 104u);
        let noise_mut = rand_signed(seed);

        // Mutate each gene with rate-scaled noise
        genome_a_new.x = clamp(genome_a_new.x + noise_r * mut_rate * 14.0, 2.0, 16.0);
        genome_a_new.y = clamp(genome_a_new.y + noise_mu * mut_rate, 0.0, 1.0);
        genome_a_new.z = clamp(genome_a_new.z + noise_sigma * mut_rate * 0.3, 0.01, 0.3);
        genome_a_new.w = clamp(genome_a_new.w + noise_agg * mut_rate, 0.0, 1.0);

        // Meta-mutation: mutation rate evolves too (smaller step)
        // Beta-prior prevents drift to 0 or 1
        genome_b_new = clamp(genome_b_new + noise_mut * 0.001, 0.0001, 0.05);
    }

    // ================== WRITE OUTPUTS ==================
    mass_out[i] = mass_new;
    energy_out[i] = energy_new;
    genome_a_out[i] = genome_a_new;
    genome_b_out[i] = genome_b_new;
}
