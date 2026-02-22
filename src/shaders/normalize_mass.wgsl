// ============================================================================
// normalize_mass.wgsl — EvoLenia v2
// Two-pass mass conservation correction.
//
// Pass A (sum_mass): Each workgroup atomically accumulates its local mass
//   into a global atomic counter.
// Pass B (normalize_mass): A single correction factor is applied globally
//   so that total mass returns to the target value.
//
// Biology: This enforces the conservation law — mass is neither created
// nor destroyed, only redistributed. This is the thermodynamic foundation
// of the ecosystem.
// ============================================================================

// --- Pass A: Parallel reduction to compute total mass ---

struct Params {
    width: u32,
    height: u32,
    target_mass_x1000: u32, // target mass * 1000, encoded as u32
    damping_x1000: u32,    // damping factor * 1000
    enabled: u32,          // 0 = disabled, 1 = enabled
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> mass: array<f32>;
@group(0) @binding(2) var<storage, read_write> mass_sum: array<atomic<u32>>;
// mass_sum[0] = accumulated total mass * 1000 (integer atomics)
// mass_sum[1] = pixel count (for normalization)

@compute @workgroup_size(256)
fn sum_mass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_pixels = params.width * params.height;
    if (gid.x >= total_pixels) {
        return;
    }

    // Atomically add mass * 1000 (integer representation for atomics)
    let m = mass[gid.x];
    let m_int = u32(m * 1000.0);
    atomicAdd(&mass_sum[0], m_int);
}

// --- Pass B: Apply correction factor to all pixels ---

@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_pixels = params.width * params.height;
    if (gid.x >= total_pixels) {
        return;
    }

    let actual_total = f32(atomicLoad(&mass_sum[0])) / 1000.0;
    let target_total = f32(params.target_mass_x1000) / 1000.0;

    if (params.enabled > 0u && actual_total > 0.001) {
        let raw_correction = target_total / actual_total;
        // Soft correction: blend toward target with damping factor (parameterized)
        let damping = f32(params.damping_x1000) / 1000.0;
        let correction = 1.0 + (raw_correction - 1.0) * damping;
        let corrected = clamp(mass[gid.x] * correction, 0.0, 1.0);
        mass[gid.x] = corrected;
    }
}
