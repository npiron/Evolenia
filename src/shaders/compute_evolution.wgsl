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
    mutation_rate_mult: f32,
    predation_factor: f32,
    radius_cost_exp: f32,      // exponent for radius metabolic cost
    agg_mobility: f32,         // aggressivity-mobility tradeoff strength
    starvation_severity: f32,  // mass decay multiplier when starving
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
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
    // Guard: sigma must be > 0 to avoid division by zero in growth function
    // exp(-x²/(2·0²)) = exp(-∞) or exp(NaN) — catastrophic for the simulation
    let sigma  = max(ga.z, 0.005); // growth width (tolerance), minimum 0.005
    let agg    = ga.w; // aggressivity

    // Base seed for PRNG — unique per pixel per frame
    let base_seed = (gid.y * params.width + gid.x) ^ params.frame ^ 0xDEADBEEFu;

    // ================== EARLY EXIT FOR EMPTY REGIONS ==================
    // Skip the expensive convolution for dead pixels with no living
    // neighbors within kernel range. With ~85% fill=0 this skips most work.
    if (m < 0.001) {
        let max_r_i = 14; // must be >= max_r used in convolution
        let hr = max_r_i / 2;
        // Sparse 16-point check: cardinal at max_r, diagonals at half, immediate neighbors, mid-ring
        let has_life =
            mass_in[idx(x + max_r_i, y)] > 0.001 ||
            mass_in[idx(x - max_r_i, y)] > 0.001 ||
            mass_in[idx(x, y + max_r_i)] > 0.001 ||
            mass_in[idx(x, y - max_r_i)] > 0.001 ||
            mass_in[idx(x + hr, y + hr)] > 0.001 ||
            mass_in[idx(x - hr, y + hr)] > 0.001 ||
            mass_in[idx(x + hr, y - hr)] > 0.001 ||
            mass_in[idx(x - hr, y - hr)] > 0.001 ||
            mass_in[idx(x + hr, y)] > 0.001 ||
            mass_in[idx(x - hr, y)] > 0.001 ||
            mass_in[idx(x, y + hr)] > 0.001 ||
            mass_in[idx(x, y - hr)] > 0.001 ||
            mass_in[idx(x + 1, y)] > 0.001 ||
            mass_in[idx(x - 1, y)] > 0.001 ||
            mass_in[idx(x, y + 1)] > 0.001 ||
            mass_in[idx(x, y - 1)] > 0.001;

        if (!has_life) {
            mass_out[i] = 0.0;
            energy_out[i] = e;
            genome_a_out[i] = ga;
            genome_b_out[i] = gb;
            return;
        }
    }

    // ================== LENIA CONVOLUTION ==================
    // Four-tier kernel interpolation supporting radii from 3 to 15.
    // max_r=13 enables proper Lenia patterns (orbium, geminium, etc.)
    // which require effective radii of 10-15 pixels.
    // 27×27 = 729 samples per pixel — fast enough on modern GPUs.
    let r_small  = 3.0;
    let r_mid    = 6.0;
    let r_large  = 10.0;
    let r_xlarge = 15.0;
    let max_r    = 13;  // 27×27 convolution — required for Lenia creatures

    var U = 0.0; // Perceived density (convolution result)
    var kernel_sum = 0.0;

    // Sample the neighborhood up to max kernel radius
    for (var dy = -max_r; dy <= max_r; dy = dy + 1) {
        for (var dx = -max_r; dx <= max_r; dx = dx + 1) {
            let dist = sqrt(f32(dx * dx + dy * dy));
            if (dist < 0.5 || dist > f32(max_r)) {
                continue;
            }

            // Four-tier smooth interpolation based on genome radius r
            var w = 0.0;
            if (r <= r_small) {
                w = kernel_weight(dist, r_small);
                if (dist > r_small * 1.5) { w = 0.0; }
            } else if (r <= r_mid) {
                let t = (r - r_small) / (r_mid - r_small);
                let w_s = kernel_weight(dist, r_small);
                let w_m = kernel_weight(dist, r_mid);
                let ws_faded = select(0.0, w_s, dist <= r_small * 1.5);
                w = mix(ws_faded, w_m, t);
                if (dist > r_mid * 1.5) { w *= (1.0 - t); }
            } else if (r <= r_large) {
                let t = (r - r_mid) / (r_large - r_mid);
                let w_m = kernel_weight(dist, r_mid);
                let w_l = kernel_weight(dist, r_large);
                let wm_faded = select(0.0, w_m, dist <= r_mid * 1.5);
                w = mix(wm_faded, w_l, t);
                if (dist > r_large * 1.5) { w *= (1.0 - t); }
            } else {
                let t = clamp((r - r_large) / (r_xlarge - r_large), 0.0, 1.0);
                let w_l = kernel_weight(dist, r_large);
                let w_x = kernel_weight(dist, r_xlarge);
                let wl_faded = select(0.0, w_l, dist <= r_large * 1.5);
                w = mix(wl_faded, w_x, t);
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
    // Costs reduced vs v1 so Lenia-scale creatures (R=10-15) can survive.
    // Non-linear radius cost: pow(r/15, exponent) — normalized to max radius 15
    let genomic_complexity = length(vec3<f32>(mu, sigma, agg));
    let radius_penalty = pow(r / 15.0, params.radius_cost_exp) * 0.02;
    let agg_penalty = agg * agg * 0.03 * params.predation_factor;
    let predator_interference = agg * agg * agg * 0.015 * params.predation_factor;
    let cost = (genomic_complexity * 0.012 + radius_penalty + agg_penalty + predator_interference) * m;
    // Absorption from local resource map (nutrient uptake)
    // Increased absorption to support larger organisms with bigger radii
    let prey_bonus = (1.0 - agg) * 0.010;
    let absorption = resource_map[i] * m * (0.040 + prey_bonus);
    var energy_new = clamp(e + absorption - cost, 0.0, 1.0);

    // Starvation: significant mass decay when energy depleted
    if (energy_new <= 0.05) {
        let starvation_k = 1.0 - energy_new / 0.05; // 0 at e=0.05, 1 at e=0
        mass_candidate *= 1.0 - params.starvation_severity * starvation_k;
    }

    // ================== MASS-CONSERVATIVE ADVECTION ==================
    // Mass is TRANSFERRED, never copied. Conservation: flux_in = flux_out
    let vel = velocity[i];

    // Cardinal direction vectors
    var total_flux_out = 0.0;
    var total_flux_in = 0.0;

    // Flux limiters — unrolled (WGSL requires constant indices for local arrays)
    // Cap per direction = mass/8 (not /4): prevents >50% total outflow per step
    // right
    { let fc = dot(vel, vec2<f32>(1.0, 0.0)); total_flux_out += clamp(fc, 0.0, mass_candidate / 8.0);
      let ni = idx(x + 1, y); let vn = velocity[ni]; let mn = mass_in[ni];
      let fi = dot(vn, vec2<f32>(-1.0, 0.0)); total_flux_in += clamp(fi, 0.0, mn / 8.0); }
    // left
    { let fc = dot(vel, vec2<f32>(-1.0, 0.0)); total_flux_out += clamp(fc, 0.0, mass_candidate / 8.0);
      let ni = idx(x - 1, y); let vn = velocity[ni]; let mn = mass_in[ni];
      let fi = dot(vn, vec2<f32>(1.0, 0.0)); total_flux_in += clamp(fi, 0.0, mn / 8.0); }
    // down
    { let fc = dot(vel, vec2<f32>(0.0, 1.0)); total_flux_out += clamp(fc, 0.0, mass_candidate / 8.0);
      let ni = idx(x, y + 1); let vn = velocity[ni]; let mn = mass_in[ni];
      let fi = dot(vn, vec2<f32>(0.0, -1.0)); total_flux_in += clamp(fi, 0.0, mn / 8.0); }
    // up
    { let fc = dot(vel, vec2<f32>(0.0, -1.0)); total_flux_out += clamp(fc, 0.0, mass_candidate / 8.0);
      let ni = idx(x, y - 1); let vn = velocity[ni]; let mn = mass_in[ni];
      let fi = dot(vn, vec2<f32>(0.0, 1.0)); total_flux_in += clamp(fi, 0.0, mn / 8.0); }

    var mass_new = mass_candidate + total_flux_in - total_flux_out;
    mass_new = clamp(mass_new, 0.0, 1.0);

    // ================== DNA ADVECTION — STOCHASTIC SEGREGATION ==================
    // When mass flows from neighbor to self, the neighbor's genome can
    // "colonize" this cell. Probability proportional to flux/mass ratio.
    // Biology: this implements spatial heredity via mass transport.
    var genome_a_new = ga;
    var genome_b_new = gb;

    var seed = base_seed;
    // Genome advection — unrolled
    // right
    { let ni = idx(x + 1, y); let vn = velocity[ni]; let mn = mass_in[ni];
      let fi = clamp(dot(vn, vec2<f32>(-1.0, 0.0)), 0.0, mn / 4.0);
      if (fi > 0.001) { let p = fi / (mass_new + 0.001); seed = pcg_hash(seed + 1u);
        if (rand01(seed) < p) { genome_a_new = genome_a_in[ni]; genome_b_new = genome_b_in[ni]; } } }
    // left
    { let ni = idx(x - 1, y); let vn = velocity[ni]; let mn = mass_in[ni];
      let fi = clamp(dot(vn, vec2<f32>(1.0, 0.0)), 0.0, mn / 4.0);
      if (fi > 0.001) { let p = fi / (mass_new + 0.001); seed = pcg_hash(seed + 2u);
        if (rand01(seed) < p) { genome_a_new = genome_a_in[ni]; genome_b_new = genome_b_in[ni]; } } }
    // down
    { let ni = idx(x, y + 1); let vn = velocity[ni]; let mn = mass_in[ni];
      let fi = clamp(dot(vn, vec2<f32>(0.0, -1.0)), 0.0, mn / 4.0);
      if (fi > 0.001) { let p = fi / (mass_new + 0.001); seed = pcg_hash(seed + 3u);
        if (rand01(seed) < p) { genome_a_new = genome_a_in[ni]; genome_b_new = genome_b_in[ni]; } } }
    // up
    { let ni = idx(x, y - 1); let vn = velocity[ni]; let mn = mass_in[ni];
      let fi = clamp(dot(vn, vec2<f32>(0.0, 1.0)), 0.0, mn / 4.0);
      if (fi > 0.001) { let p = fi / (mass_new + 0.001); seed = pcg_hash(seed + 4u);
        if (rand01(seed) < p) { genome_a_new = genome_a_in[ni]; genome_b_new = genome_b_in[ni]; } } }

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

        // Mutate each gene with rate-scaled noise — smaller steps to preserve Lenia patterns
        let mm = params.mutation_rate_mult;
        genome_a_new.x = clamp(genome_a_new.x + noise_r     * mut_rate * mm * 3.0,  3.0, 15.0);
        genome_a_new.y = clamp(genome_a_new.y + noise_mu    * mut_rate * mm * 0.15, 0.05, 0.35);
        genome_a_new.z = clamp(genome_a_new.z + noise_sigma * mut_rate * mm * 0.08, 0.005, 0.08);
        genome_a_new.w = clamp(genome_a_new.w + noise_agg   * mut_rate * mm * 0.3,  0.0, 1.0);

        // Meta-mutation: mutation rate evolves too (smaller step)
        // Beta-prior prevents drift to 0 or 1
        genome_b_new = clamp(genome_b_new + noise_mut * mm * 0.0002, 0.0005, 0.008);
    }

    // ================== GENOME CONSENSUS (spatial coherence) ==================
    // Mass-weighted blending with immediate neighbors — creates coherent
    // "organism" regions where nearby pixels share similar genomes.
    // Without this, per-pixel mutations fragment genome into noise.
    // Biologically: horizontal gene transfer / developmental coherence.
    if (mass_new > 0.01) {
        let blend_strength = 0.08; // subtle but cumulative over frames
        var neighbor_genome_a = vec4<f32>(0.0);
        var neighbor_genome_b = 0.0;
        var neighbor_weight = 0.0;

        // 4-connected neighbors, weighted by their mass
        let nr = idx(x + 1, y); let mr = mass_in[nr];
        let nl = idx(x - 1, y); let ml = mass_in[nl];
        let nd = idx(x, y + 1); let md = mass_in[nd];
        let nu = idx(x, y - 1); let mu_n = mass_in[nu];

        neighbor_genome_a += genome_a_in[nr] * mr;
        neighbor_genome_a += genome_a_in[nl] * ml;
        neighbor_genome_a += genome_a_in[nd] * md;
        neighbor_genome_a += genome_a_in[nu] * mu_n;
        neighbor_genome_b += genome_b_in[nr] * mr;
        neighbor_genome_b += genome_b_in[nl] * ml;
        neighbor_genome_b += genome_b_in[nd] * md;
        neighbor_genome_b += genome_b_in[nu] * mu_n;
        neighbor_weight = mr + ml + md + mu_n;

        if (neighbor_weight > 0.01) {
            let avg_ga = neighbor_genome_a / neighbor_weight;
            let avg_gb = neighbor_genome_b / neighbor_weight;
            genome_a_new = mix(genome_a_new, avg_ga, blend_strength);
            genome_b_new = mix(genome_b_new, avg_gb, blend_strength);
        }
    }

    // ================== WRITE OUTPUTS ==================
    mass_out[i] = mass_new;
    energy_out[i] = energy_new;
    genome_a_out[i] = genome_a_new;
    genome_b_out[i] = genome_b_new;
}
