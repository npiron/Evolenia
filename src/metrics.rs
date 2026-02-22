// ============================================================================
// metrics.rs — EvoLenia v2
// Emergence metrics: genetic diversity, species detection, entropy calculation.
// GPU readback diagnostics for comprehensive simulation monitoring.
// ============================================================================

use std::collections::HashMap;

use crate::world::BufferSnapshot;

// ======================== Full Diagnostics Report ========================

/// Complete diagnostics snapshot for one frame.
pub struct SimDiagnostics {
    // Population
    pub total_mass: f32,
    pub live_pixels: u32,
    pub live_fraction: f32,
    pub max_mass: f32,
    pub avg_mass_live: f32, // average mass over live pixels only

    // Energy
    pub avg_energy: f32,     // over live pixels
    pub min_energy_live: f32,
    pub starving_fraction: f32, // fraction of live pixels with energy ≤ 0.01

    // Resources
    pub avg_resource: f32,
    pub min_resource: f32,
    pub depleted_fraction: f32, // fraction of pixels with resource < 0.1

    // Genetics
    pub genetic_entropy: f32,
    pub species_count: usize,
    pub genome_stats: GenomeStats,

    // Spatial
    pub mass_std_dev: f32, // spatial uniformity of mass
}

impl SimDiagnostics {
    /// Compute full diagnostics from a GPU readback snapshot.
    pub fn from_snapshot(snap: &BufferSnapshot) -> Self {
        let n = snap.mass.len();

        // ---- Population stats ----
        let mut total_mass = 0.0f64;
        let mut live_pixels = 0u32;
        let mut max_mass = 0.0f32;
        let mut sum_energy = 0.0f64;
        let mut min_energy_live = 1.0f32;
        let mut starving = 0u32;

        for i in 0..n {
            let m = snap.mass[i];
            total_mass += m as f64;
            if m > max_mass { max_mass = m; }
            if m > 0.01 {
                live_pixels += 1;
                let e = snap.energy[i];
                sum_energy += e as f64;
                if e < min_energy_live { min_energy_live = e; }
                if e <= 0.01 { starving += 1; }
            }
        }

        let live_fraction = live_pixels as f32 / n as f32;
        let avg_mass_live = if live_pixels > 0 { total_mass as f32 / live_pixels as f32 } else { 0.0 };
        let avg_energy = if live_pixels > 0 { sum_energy as f32 / live_pixels as f32 } else { 0.0 };
        let starving_fraction = if live_pixels > 0 { starving as f32 / live_pixels as f32 } else { 0.0 };

        // ---- Mass spatial std dev ----
        let mean_mass = total_mass as f32 / n as f32;
        let mut var = 0.0f64;
        for i in 0..n {
            let diff = snap.mass[i] - mean_mass;
            var += (diff * diff) as f64;
        }
        let mass_std_dev = (var / n as f64).sqrt() as f32;

        // ---- Resource stats ----
        let mut sum_resource = 0.0f64;
        let mut min_resource = 1.0f32;
        let mut depleted = 0u32;
        for i in 0..n {
            let r = snap.resource[i];
            sum_resource += r as f64;
            if r < min_resource { min_resource = r; }
            if r < 0.1 { depleted += 1; }
        }
        let avg_resource = sum_resource as f32 / n as f32;
        let depleted_fraction = depleted as f32 / n as f32;

        // ---- Genetics ----
        let genetic_entropy = compute_genetic_entropy(&snap.genome_a, &snap.mass, 10);
        let species_count = detect_species(&snap.genome_a, &snap.mass, 20);
        let genome_stats = compute_genome_stats(&snap.genome_a, &snap.genome_b, &snap.mass);

        SimDiagnostics {
            total_mass: total_mass as f32,
            live_pixels,
            live_fraction,
            max_mass,
            avg_mass_live,
            avg_energy,
            min_energy_live,
            starving_fraction,
            avg_resource,
            min_resource,
            depleted_fraction,
            genetic_entropy,
            species_count,
            genome_stats,
            mass_std_dev,
        }
    }

    /// Log all diagnostics at INFO level, with optional delta from previous snapshot.
    pub fn log(&self, frame: u32, target_mass: f32, prev: Option<&SimDiagnostics>) {
        log::info!(
            "══════════════ Frame {} Diagnostics ══════════════",
            frame
        );

        if let Some(p) = prev {
            let dm = self.total_mass - p.total_mass;
            let dlive = self.live_pixels as i32 - p.live_pixels as i32;
            let dmu = self.genome_stats.avg_mu - p.genome_stats.avg_mu;
            let dagg = self.genome_stats.avg_aggressivity - p.genome_stats.avg_aggressivity;
            log::info!(
                "TRENDS: Δmass={:+.0} | Δlive={:+} | Δmu={:+.4} | Δagg={:+.4} | Δentropy={:+.2}",
                dm, dlive, dmu, dagg,
                self.genetic_entropy - p.genetic_entropy,
            );
        }

        log::info!(
            "POPULATION: mass={:.0}/{:.0} ({:.1}%) | live={} ({:.1}%) | max_m={:.3} | avg_m_live={:.3}",
            self.total_mass,
            target_mass,
            self.total_mass / target_mass * 100.0,
            self.live_pixels,
            self.live_fraction * 100.0,
            self.max_mass,
            self.avg_mass_live,
        );
        log::info!(
            "ENERGY: avg={:.3} | min_live={:.3} | starving={:.1}%",
            self.avg_energy,
            self.min_energy_live,
            self.starving_fraction * 100.0,
        );
        log::info!(
            "RESOURCES: avg={:.3} | min={:.3} | depleted(<0.1)={:.1}%",
            self.avg_resource,
            self.min_resource,
            self.depleted_fraction * 100.0,
        );
        log::info!(
            "GENETICS: entropy={:.2} bits | species={} | predators={:.1}%",
            self.genetic_entropy,
            self.species_count,
            self.genome_stats.predator_fraction * 100.0,
        );
        log::info!(
            "GENOME AVG: r={:.2} mu={:.3} sigma={:.3} agg={:.3} mut_rate={:.5}",
            self.genome_stats.avg_radius,
            self.genome_stats.avg_mu,
            self.genome_stats.avg_sigma,
            self.genome_stats.avg_aggressivity,
            self.genome_stats.avg_mutation_rate,
        );
        log::info!(
            "SPATIAL: mass_stddev={:.4}",
            self.mass_std_dev,
        );
    }
}

// ======================== Genetic Entropy ========================

/// Computes Shannon entropy of genome distribution.
/// Higher entropy = more genetic diversity.
///
/// H = -Σ p(i) * log2(p(i))
/// where p(i) is the probability of genome bin i.
pub fn compute_genetic_entropy(genome_a: &[f32], mass: &[f32], bins: usize) -> f32 {
    if genome_a.len() < 4 || mass.is_empty() {
        return 0.0;
    }

    let num_pixels = genome_a.len() / 4;
    let mut histogram: HashMap<(u8, u8, u8), f32> = HashMap::new();
    let mut total_mass = 0.0;

    // Bin each genome by (r, mu, sigma) discretized to bins
    for i in 0..num_pixels {
        let m = mass[i];
        if m < 0.01 {
            continue; // Skip dead pixels
        }

        let r = genome_a[i * 4];
        let mu = genome_a[i * 4 + 1];
        let sigma = genome_a[i * 4 + 2];

        // Discretize to bins (0..bins-1)
        let r_bin = ((r / 16.0) * bins as f32).min((bins - 1) as f32) as u8;
        let mu_bin = (mu * bins as f32).min((bins - 1) as f32) as u8;
        let sigma_bin = ((sigma / 0.3) * bins as f32).min((bins - 1) as f32) as u8;

        let key = (r_bin, mu_bin, sigma_bin);
        *histogram.entry(key).or_insert(0.0) += m;
        total_mass += m;
    }

    if total_mass < 1e-6 {
        return 0.0;
    }

    // Compute Shannon entropy
    let mut entropy = 0.0;
    for &count in histogram.values() {
        let p = count / total_mass;
        if p > 1e-9 {
            entropy -= p * p.log2();
        }
    }

    entropy
}

// ======================== Species Detection (k-means) ========================

/// Simple k-means clustering on genome space to detect distinct species.
/// Returns the number of clusters (species) found.
///
/// This is a simplified version. For production, use a proper clustering library.
pub fn detect_species(genome_a: &[f32], mass: &[f32], max_species: usize) -> usize {
    if genome_a.len() < 4 || mass.is_empty() {
        return 0;
    }

    let num_pixels = genome_a.len() / 4;
    
    // Collect genomes weighted by mass (alive organisms only)
    let mut genomes: Vec<(f32, f32, f32, f32)> = Vec::new();
    for i in 0..num_pixels {
        let m = mass[i];
        if m > 0.05 {
            // Only consider organisms with significant mass
            let r = genome_a[i * 4];
            let mu = genome_a[i * 4 + 1];
            let sigma = genome_a[i * 4 + 2];
            let agg = genome_a[i * 4 + 3];
            genomes.push((r, mu, sigma, agg));
        }
    }

    if genomes.len() < max_species {
        return genomes.len();
    }

    // Simple heuristic: count distinct genome clusters by variance threshold
    // Real k-means would be better but requires iterative optimization
    let mut unique_genomes: Vec<(f32, f32, f32, f32)> = Vec::new();
    let threshold = 0.15; // Genomes closer than this are considered same species

    for genome in genomes {
        let mut is_unique = true;
        for &existing in &unique_genomes {
            let dist = genome_distance(genome, existing);
            if dist < threshold {
                is_unique = false;
                break;
            }
        }
        if is_unique {
            unique_genomes.push(genome);
        }
        if unique_genomes.len() >= max_species {
            break;
        }
    }

    unique_genomes.len()
}

/// Euclidean distance in normalized genome space
fn genome_distance(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let dr = (a.0 / 16.0 - b.0 / 16.0).powi(2);
    let dmu = (a.1 - b.1).powi(2);
    let dsigma = (a.2 / 0.3 - b.2 / 0.3).powi(2);
    let dagg = (a.3 - b.3).powi(2);
    (dr + dmu + dsigma + dagg).sqrt()
}

// ======================== Genome Statistics ========================

pub struct GenomeStats {
    pub avg_radius: f32,
    pub avg_mu: f32,
    pub avg_sigma: f32,
    pub avg_aggressivity: f32,
    pub avg_mutation_rate: f32,
    pub predator_fraction: f32, // fraction with agg > 0.7
}

/// Computes mass-weighted average genome statistics
pub fn compute_genome_stats(
    genome_a: &[f32],
    genome_b: &[f32],
    mass: &[f32],
) -> GenomeStats {
    let num_pixels = genome_a.len() / 4;
    let mut total_mass = 0.0;
    let mut sum_r = 0.0;
    let mut sum_mu = 0.0;
    let mut sum_sigma = 0.0;
    let mut sum_agg = 0.0;
    let mut sum_mut = 0.0;
    let mut predator_mass = 0.0;

    for i in 0..num_pixels {
        let m = mass[i];
        if m < 0.01 {
            continue;
        }

        total_mass += m;
        sum_r += genome_a[i * 4] * m;
        sum_mu += genome_a[i * 4 + 1] * m;
        sum_sigma += genome_a[i * 4 + 2] * m;
        sum_agg += genome_a[i * 4 + 3] * m;
        sum_mut += genome_b[i] * m;

        if genome_a[i * 4 + 3] > 0.7 {
            predator_mass += m;
        }
    }

    if total_mass < 1e-6 {
        return GenomeStats {
            avg_radius: 0.0,
            avg_mu: 0.0,
            avg_sigma: 0.0,
            avg_aggressivity: 0.0,
            avg_mutation_rate: 0.0,
            predator_fraction: 0.0,
        };
    }

    GenomeStats {
        avg_radius: sum_r / total_mass,
        avg_mu: sum_mu / total_mass,
        avg_sigma: sum_sigma / total_mass,
        avg_aggressivity: sum_agg / total_mass,
        avg_mutation_rate: sum_mut / total_mass,
        predator_fraction: predator_mass / total_mass,
    }
}
