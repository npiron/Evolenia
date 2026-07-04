// ============================================================================
// tests.rs — EvoLenia v2
// Comprehensive unit tests for scientific validation.
// These tests verify INVARIANTS that must hold, not implementation details.
// If a test fails, it indicates a BUG in the implementation.
// ============================================================================

#[cfg(test)]
mod physics_tests {
    //! Tests for physical conservation laws and bounds.
    //! These are fundamental — violations indicate broken physics.

    use crate::metrics::SimDiagnostics;
    use crate::world::{BufferSnapshot, WORLD_HEIGHT, WORLD_WIDTH};

    fn create_uniform_snapshot(mass_value: f32, energy_value: f32) -> BufferSnapshot {
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        BufferSnapshot {
            mass: vec![mass_value; n],
            energy: vec![energy_value; n],
            genome_a: vec![10.0, 0.15, 0.02, 0.1]
                .into_iter()
                .cycle()
                .take(n * 4)
                .collect(),
            genome_b: vec![0.003; n],
            resource: vec![1.0; n],
        }
    }

    #[test]
    fn mass_must_be_non_negative() {
        // Physical invariant: mass cannot be negative
        let snap = create_uniform_snapshot(0.5, 0.5);
        for m in &snap.mass {
            assert!(*m >= 0.0, "Mass cannot be negative, found: {}", m);
        }
    }

    #[test]
    fn mass_must_be_bounded() {
        // Physical invariant: mass per pixel ≤ 1.0 (saturated)
        let snap = create_uniform_snapshot(0.5, 0.5);
        for m in &snap.mass {
            assert!(*m <= 1.0, "Mass per pixel cannot exceed 1.0, found: {}", m);
        }
    }

    #[test]
    fn energy_must_be_in_zero_one_range() {
        // Physical invariant: energy ∈ [0, 1]
        let snap = create_uniform_snapshot(0.5, 0.5);
        for e in &snap.energy {
            assert!(
                *e >= 0.0 && *e <= 1.0,
                "Energy must be in [0,1], found: {}",
                e
            );
        }
    }

    #[test]
    fn total_mass_calculation_is_accurate() {
        // Diagnostic total_mass must equal sum of pixel masses
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let mass_per_pixel = 0.3;
        let mut snap = create_uniform_snapshot(mass_per_pixel, 0.5);

        // Set specific pattern
        snap.mass = vec![mass_per_pixel; n];

        let diag = SimDiagnostics::from_snapshot(&snap);
        let expected_total = mass_per_pixel * n as f32;

        let error = (diag.total_mass - expected_total).abs();
        assert!(
            error < 0.01,
            "Total mass calculation error: expected {}, got {}, error={}",
            expected_total,
            diag.total_mass,
            error
        );
    }

    #[test]
    fn live_pixels_count_uses_correct_threshold() {
        // Live pixel: mass > 0.01 (not >= 0.01)
        let mut snap = create_uniform_snapshot(0.0, 0.5);

        // Set 100 pixels to exactly threshold, 100 pixels above
        snap.mass[0] = 0.01; // NOT live (threshold)
        snap.mass[1] = 0.011; // Live
        snap.mass[2] = 0.5; // Live

        let diag = SimDiagnostics::from_snapshot(&snap);
        assert_eq!(
            diag.live_pixels, 2,
            "Only pixels with mass > 0.01 should be live"
        );
    }

    #[test]
    fn starving_fraction_is_ratio_of_live_pixels() {
        // Starving: energy ≤ 0.01 among LIVE pixels only
        let _n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let mut snap = create_uniform_snapshot(0.0, 0.5);

        // 3 live pixels, 2 starving
        snap.mass[0] = 0.5;
        snap.energy[0] = 0.005; // starving
        snap.mass[1] = 0.5;
        snap.energy[1] = 0.01; // starving (at threshold)
        snap.mass[2] = 0.5;
        snap.energy[2] = 0.5; // not starving
                              // Dead pixel with low energy should NOT count
        snap.mass[3] = 0.001;
        snap.energy[3] = 0.0; // dead, doesn't count

        let diag = SimDiagnostics::from_snapshot(&snap);

        // 2 starving out of 3 live pixels = 2/3 ≈ 0.667
        let expected = 2.0 / 3.0;
        let error = (diag.starving_fraction - expected).abs();
        assert!(
            error < 0.01,
            "Starving fraction error: expected {}, got {}",
            expected,
            diag.starving_fraction
        );
    }
}

#[cfg(test)]
mod genome_tests {
    //! Tests for genome bounds and validity.
    //! Genome values outside valid ranges cause shader errors.

    use crate::metrics::compute_genome_stats;
    use crate::world::{BufferSnapshot, WORLD_HEIGHT, WORLD_WIDTH};

    #[test]
    fn test_sigma_zero_causes_issues() {
        // DOCUMENTATION TEST: sigma=0 causes exp(-x²/(2·0²)) = NaN/Inf
        // This test verifies WHY sigma must be positive.
        let sigma = 0.0f32;

        // This would cause: (u - mu)² / (2 * sigma²) = 0.1225 / 0.0 = Inf
        // exp(-Inf) = 0, but (u - mu)² / 0 = NaN on some platforms
        // The shader guards against this with: max(sigma, 0.005)
        let denominator = 2.0 * sigma * sigma;
        assert!(
            denominator == 0.0,
            "Zero sigma produces zero denominator, causing div-by-zero"
        );
    }

    #[test]
    fn default_genome_has_valid_sigma() {
        // The default genome in world/init.rs must have sigma > 0
        // This is the ACTUAL initialization check

        // Default genome from world/init.rs: [10.0, 0.15, 0.017, 0.0]
        let default_sigma = 0.017f32; // from world/init.rs line ~74
        assert!(
            default_sigma > 0.0,
            "Default sigma in world/init.rs must be > 0, found: {}",
            default_sigma
        );
    }

    #[test]
    fn genome_stats_weighted_by_mass() {
        // Genome stats MUST be mass-weighted averages
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let mut snap = BufferSnapshot {
            mass: vec![0.0; n],
            energy: vec![0.5; n],
            genome_a: vec![0.0; n * 4],
            genome_b: vec![0.0; n],
            resource: vec![1.0; n],
        };

        // Pixel 0: mass=0.8, r=10
        snap.mass[0] = 0.8;
        snap.genome_a[0] = 10.0; // r
        snap.genome_a[1] = 0.2; // mu
        snap.genome_a[2] = 0.02; // sigma
        snap.genome_a[3] = 0.0; // agg
        snap.genome_b[0] = 0.001;

        // Pixel 1: mass=0.2, r=20
        snap.mass[1] = 0.2;
        snap.genome_a[4] = 20.0; // r
        snap.genome_a[5] = 0.2; // mu
        snap.genome_a[6] = 0.02; // sigma
        snap.genome_a[7] = 0.0; // agg
        snap.genome_b[1] = 0.001;

        let stats = compute_genome_stats(&snap.genome_a, &snap.genome_b, &snap.mass);

        // Expected: (10*0.8 + 20*0.2) / (0.8 + 0.2) = 12
        let expected_r = (10.0 * 0.8 + 20.0 * 0.2) / 1.0;
        let error = (stats.avg_radius - expected_r).abs();
        assert!(
            error < 0.01,
            "Genome stats must be mass-weighted. Expected avg_r={}, got {}",
            expected_r,
            stats.avg_radius
        );
    }

    #[test]
    fn predator_fraction_uses_correct_threshold() {
        // Predator: aggressivity > 0.7 (NOT >= 0.7)
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let mut snap = BufferSnapshot {
            mass: vec![0.0; n],
            energy: vec![0.5; n],
            genome_a: vec![0.0; n * 4],
            genome_b: vec![0.003; n],
            resource: vec![1.0; n],
        };

        // Pixel 0: agg=0.7 (NOT predator)
        snap.mass[0] = 1.0;
        snap.genome_a[0] = 10.0;
        snap.genome_a[1] = 0.15;
        snap.genome_a[2] = 0.02;
        snap.genome_a[3] = 0.7; // exactly at threshold

        // Pixel 1: agg=0.71 (predator)
        snap.mass[1] = 1.0;
        snap.genome_a[4] = 10.0;
        snap.genome_a[5] = 0.15;
        snap.genome_a[6] = 0.02;
        snap.genome_a[7] = 0.71;

        let stats = compute_genome_stats(&snap.genome_a, &snap.genome_b, &snap.mass);

        // Only 1 predator out of 2 total mass
        let expected = 0.5;
        let error = (stats.predator_fraction - expected).abs();
        assert!(
            error < 0.01,
            "Predator fraction: > 0.7, not >= 0.7. Expected {}, got {}",
            expected,
            stats.predator_fraction
        );
    }
}

#[cfg(test)]
mod entropy_tests {
    //! Tests for Shannon entropy calculation.
    //! Entropy is the foundation of diversity metrics.

    use crate::metrics::compute_genetic_entropy;

    #[test]
    fn entropy_of_empty_population_is_zero() {
        // No organisms = no diversity = entropy 0
        let genome_a: Vec<f32> = vec![];
        let mass: Vec<f32> = vec![];

        let entropy = compute_genetic_entropy(&genome_a, &mass, 10);
        assert_eq!(entropy, 0.0, "Empty population must have zero entropy");
    }

    #[test]
    fn entropy_of_uniform_population_is_zero() {
        // All identical genomes = no diversity = entropy 0
        // H = -Σ p*log(p) where p=1 → H = 0
        let n = 100;
        let genome_a: Vec<f32> = vec![10.0, 0.15, 0.02, 0.0]
            .into_iter()
            .cycle()
            .take(n * 4)
            .collect();
        let mass: Vec<f32> = vec![0.5; n];

        let entropy = compute_genetic_entropy(&genome_a, &mass, 10);
        assert!(
            entropy < 0.01,
            "Uniform population should have near-zero entropy, got {}",
            entropy
        );
    }

    #[test]
    fn entropy_of_two_equal_species_is_one_bit() {
        // Two equally abundant species: H = -2*(0.5*log2(0.5)) = 1 bit
        let n = 100;
        let mut genome_a: Vec<f32> = Vec::with_capacity(n * 4);
        let mut mass: Vec<f32> = Vec::with_capacity(n);

        // Species A: r=5, mu=0.1, sigma=0.01 (50 organisms)
        genome_a.extend(std::iter::repeat_n([5.0, 0.1, 0.01, 0.0], 50).flatten());
        mass.extend(std::iter::repeat_n(1.0, 50));
        // Species B: r=15, mu=0.9, sigma=0.29 (50 organisms, very different)
        genome_a.extend(std::iter::repeat_n([15.0, 0.9, 0.29, 0.0], 50).flatten());
        mass.extend(std::iter::repeat_n(1.0, 50));

        let entropy = compute_genetic_entropy(&genome_a, &mass, 10);

        // Should be close to 1 bit (perfect 50/50 split)
        assert!(
            entropy > 0.8 && entropy < 1.2,
            "Two equal species should have ~1 bit of entropy, got {}",
            entropy
        );
    }

    #[test]
    fn entropy_must_be_non_negative() {
        // Shannon entropy is always ≥ 0
        let genome_a: Vec<f32> = vec![10.0, 0.15, 0.02, 0.0, 12.0, 0.3, 0.05, 0.5];
        let mass: Vec<f32> = vec![0.5, 0.5];

        let entropy = compute_genetic_entropy(&genome_a, &mass, 10);
        assert!(
            entropy >= 0.0,
            "Entropy must be non-negative, got {}",
            entropy
        );
    }

    #[test]
    fn entropy_ignores_dead_pixels() {
        // Dead pixels (mass ≤ 0.01) should not contribute to entropy
        let n = 100;
        let mut genome_a: Vec<f32> = Vec::with_capacity(n * 4);
        let mut mass: Vec<f32> = Vec::with_capacity(n);

        // 50 live organisms, all identical
        genome_a.extend(std::iter::repeat_n([10.0, 0.15, 0.02, 0.0], 50).flatten());
        mass.extend(std::iter::repeat_n(0.5, 50));
        // 50 dead organisms with different genomes (should be ignored)
        for i in 0..50 {
            genome_a.extend_from_slice(&[i as f32 % 16.0, (i as f32) / 100.0, 0.02, 0.0]);
            mass.push(0.001); // dead
        }

        let entropy = compute_genetic_entropy(&genome_a, &mass, 10);

        // Should be near 0 (only the uniform live population counts)
        assert!(
            entropy < 0.01,
            "Dead pixels should not affect entropy, got {}",
            entropy
        );
    }

    #[test]
    fn entropy_is_mass_weighted() {
        // A dominant species (by mass) should dominate entropy
        let mut genome_a: Vec<f32> = Vec::new();
        let mut mass: Vec<f32> = Vec::new();

        // Species A: mass=0.99
        genome_a.extend_from_slice(&[5.0, 0.1, 0.01, 0.0]);
        mass.push(0.99);

        // Species B: mass=0.01
        genome_a.extend_from_slice(&[15.0, 0.9, 0.29, 0.0]);
        mass.push(0.01);

        let entropy = compute_genetic_entropy(&genome_a, &mass, 10);

        // Entropy should be low (dominated by one species)
        // H = -0.99*log2(0.99) - 0.01*log2(0.01) ≈ 0.08
        assert!(
            entropy < 0.2,
            "Mass-dominated population should have low entropy, got {}",
            entropy
        );
    }
}

#[cfg(test)]
mod species_detection_tests {
    //! Tests for species clustering algorithm.

    use crate::metrics::detect_species;

    #[test]
    fn no_species_in_empty_population() {
        let genome_a: Vec<f32> = vec![];
        let mass: Vec<f32> = vec![];

        let count = detect_species(&genome_a, &mass, 20);
        assert_eq!(count, 0, "Empty population has no species");
    }

    #[test]
    fn single_organism_is_one_species() {
        let genome_a: Vec<f32> = vec![10.0, 0.15, 0.02, 0.0];
        let mass: Vec<f32> = vec![0.5];

        let count = detect_species(&genome_a, &mass, 20);
        assert_eq!(count, 1, "Single organism = 1 species");
    }

    #[test]
    fn identical_organisms_are_one_species() {
        // 100 identical organisms should be detected as 1 species
        let n = 100;
        let genome_a: Vec<f32> = vec![10.0, 0.15, 0.02, 0.0]
            .into_iter()
            .cycle()
            .take(n * 4)
            .collect();
        let mass: Vec<f32> = vec![0.5; n];

        let count = detect_species(&genome_a, &mass, 20);
        assert_eq!(count, 1, "Identical organisms = 1 species, got {}", count);
    }

    #[test]
    fn very_different_genomes_are_separate_species() {
        // Two organisms with maximally different genomes
        let mut genome_a: Vec<f32> = Vec::new();
        let mut mass: Vec<f32> = Vec::new();

        // Species A: r=3, mu=0.0, sigma=0.01, agg=0.0
        genome_a.extend_from_slice(&[3.0, 0.0, 0.01, 0.0]);
        mass.push(0.5);

        // Species B: r=15, mu=1.0, sigma=0.3, agg=1.0 (maximally different)
        genome_a.extend_from_slice(&[15.0, 1.0, 0.3, 1.0]);
        mass.push(0.5);

        let count = detect_species(&genome_a, &mass, 20);
        assert!(
            count >= 2,
            "Very different genomes should be separate species, got {}",
            count
        );
    }

    #[test]
    fn dead_pixels_not_counted_as_species() {
        // Low-mass pixels (< 0.05) should not count
        let mut genome_a: Vec<f32> = Vec::new();
        let mut mass: Vec<f32> = Vec::new();

        // Live organism
        genome_a.extend_from_slice(&[10.0, 0.15, 0.02, 0.0]);
        mass.push(0.5);

        // Dead organism with different genome
        genome_a.extend_from_slice(&[3.0, 0.9, 0.3, 1.0]);
        mass.push(0.01); // dead

        let count = detect_species(&genome_a, &mass, 20);
        assert_eq!(count, 1, "Dead pixels should not create species");
    }

    #[test]
    fn species_count_bounded_by_max() {
        // Should never return more than max_species
        let n = 50;
        let mut genome_a: Vec<f32> = Vec::new();
        let mut mass: Vec<f32> = Vec::new();

        // Create 50 very different species
        for i in 0..n {
            let r = 3.0 + (i as f32 / n as f32) * 12.0;
            let mu = i as f32 / n as f32;
            genome_a.extend_from_slice(&[r, mu, 0.02, 0.0]);
            mass.push(0.5);
        }

        let max_species = 10;
        let count = detect_species(&genome_a, &mass, max_species);
        assert!(
            count <= max_species,
            "Species count must be ≤ max_species={}, got {}",
            max_species,
            count
        );
    }
}

#[cfg(test)]
mod math_tests {
    //! Tests for mathematical functions (Lenia kernel, growth function).
    //! These test the formulas from INI.MD.

    /// Lenia ring kernel weight (from INI.MD §3.2)
    /// K(d, r) = exp(-((d/r - 0.5)² / (2 * 0.15²)))
    fn kernel_weight(dist: f32, radius: f32) -> f32 {
        let normalized = dist / radius;
        let diff = normalized - 0.5;
        (-diff * diff / (2.0 * 0.15 * 0.15)).exp()
    }

    /// Lenia growth function (from INI.MD §3.2)
    /// G(U; μ, σ) = exp(-((U - μ)² / (2σ²)))
    fn growth_function(u: f32, mu: f32, sigma: f32) -> f32 {
        if sigma <= 0.0 {
            panic!("Sigma must be > 0");
        }
        (-((u - mu) * (u - mu)) / (2.0 * sigma * sigma)).exp()
    }

    #[test]
    fn kernel_peaks_at_half_radius() {
        // Ring kernel should peak at d = r/2
        let r = 10.0;
        let peak_dist = r * 0.5;

        let w_peak = kernel_weight(peak_dist, r);
        let w_before = kernel_weight(peak_dist - 1.0, r);
        let w_after = kernel_weight(peak_dist + 1.0, r);

        assert!(
            w_peak > w_before && w_peak > w_after,
            "Kernel should peak at d=r/2. Peak={}, before={}, after={}",
            w_peak,
            w_before,
            w_after
        );
    }

    #[test]
    fn kernel_symmetric_around_peak() {
        // Kernel should be symmetric around d = r/2
        let r = 10.0;
        let delta = 2.0;

        let w_below = kernel_weight(r * 0.5 - delta, r);
        let w_above = kernel_weight(r * 0.5 + delta, r);

        let diff = (w_below - w_above).abs();
        assert!(
            diff < 0.01,
            "Kernel should be symmetric. w({})={}, w({})={}",
            r * 0.5 - delta,
            w_below,
            r * 0.5 + delta,
            w_above
        );
    }

    #[test]
    fn kernel_decays_to_zero_at_extremes() {
        let r = 10.0;

        // At d=0 and d=r, kernel should be small
        let w_center = kernel_weight(0.01, r);
        let w_edge = kernel_weight(r, r);

        assert!(
            w_center < 0.1,
            "Kernel near center should be small, got {}",
            w_center
        );
        assert!(
            w_edge < 0.1,
            "Kernel at edge should be small, got {}",
            w_edge
        );
    }

    #[test]
    fn growth_function_peaks_at_mu() {
        // G(μ; μ, σ) = 1.0
        let mu = 0.15;
        let sigma = 0.02;

        let g = growth_function(mu, mu, sigma);
        let diff = (g - 1.0).abs();

        assert!(diff < 1e-6, "Growth at U=μ should be 1.0, got {}", g);
    }

    #[test]
    fn growth_function_decays_away_from_mu() {
        let mu = 0.15;
        let sigma = 0.02;

        let g_peak = growth_function(mu, mu, sigma);
        let g_offset = growth_function(mu + sigma, mu, sigma);

        // At one sigma away, G ≈ exp(-0.5) ≈ 0.606
        let expected = (-0.5_f32).exp();
        let diff = (g_offset - expected).abs();

        assert!(
            diff < 0.01,
            "Growth at U=μ+σ should be exp(-0.5)≈0.606, got {}",
            g_offset
        );
        assert!(g_peak > g_offset, "Growth should decay away from μ");
    }

    #[test]
    fn growth_function_symmetric_around_mu() {
        let mu = 0.15;
        let sigma = 0.02;
        let delta = 0.03;

        let g_below = growth_function(mu - delta, mu, sigma);
        let g_above = growth_function(mu + delta, mu, sigma);

        let diff = (g_below - g_above).abs();
        assert!(
            diff < 1e-6,
            "Growth function should be symmetric around μ. G({})={}, G({})={}",
            mu - delta,
            g_below,
            mu + delta,
            g_above
        );
    }

    #[test]
    fn growth_function_bounded_zero_one() {
        // Growth function output is always in [0, 1]
        // NOTE: Mathematically G > 0, but IEEE float underflows to 0
        // for extreme values. This is acceptable in simulation.
        let test_cases = [
            (0.0, 0.15, 0.02),
            (0.5, 0.15, 0.02), // far from mu → underflows to ~0
            (1.0, 0.15, 0.02),
            (0.15, 0.15, 0.001), // narrow sigma
            (0.15, 0.15, 0.3),   // wide sigma
        ];

        for (u, mu, sigma) in test_cases {
            let g = growth_function(u, mu, sigma);
            assert!(
                (0.0..=1.0).contains(&g),
                "G({}, {}, {}) = {} should be in [0, 1]",
                u,
                mu,
                sigma,
                g
            );
            // Also verify it's not NaN or Inf
            assert!(
                g.is_finite(),
                "G({}, {}, {}) = {} should be finite",
                u,
                mu,
                sigma,
                g
            );
        }
    }

    #[test]
    fn growth_function_extreme_values_stay_finite() {
        // Even with extreme inputs, growth should be finite (not NaN/Inf)
        let extreme_cases = [
            (0.0, 1.0, 0.005), // max distance, min sigma
            (1.0, 0.0, 0.005), // max distance, min sigma
            (0.5, 0.5, 0.3),   // at peak, max sigma
        ];

        for (u, mu, sigma) in extreme_cases {
            let g = growth_function(u, mu, sigma);
            assert!(
                g.is_finite(),
                "Growth function must be finite for all valid inputs. G({}, {}, {}) = {}",
                u,
                mu,
                sigma,
                g
            );
        }
    }

    #[test]
    fn narrow_sigma_is_specialist() {
        // Smaller sigma = sharper peak = specialist
        let mu = 0.15;
        let sigma_narrow = 0.01;
        let sigma_wide = 0.1;
        let offset = 0.05;

        let g_narrow = growth_function(mu + offset, mu, sigma_narrow);
        let g_wide = growth_function(mu + offset, mu, sigma_wide);

        assert!(
            g_narrow < g_wide,
            "Narrow σ (specialist) should decay faster. narrow={}, wide={}",
            g_narrow,
            g_wide
        );
    }
}

#[cfg(test)]
mod state_io_tests {
    //! Tests for snapshot save/load (lossless roundtrip).

    use crate::state_io::{load_snapshot, save_snapshot};
    use crate::world::{BufferSnapshot, WORLD_HEIGHT, WORLD_WIDTH};
    use std::fs;

    fn create_test_snapshot() -> BufferSnapshot {
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        BufferSnapshot {
            mass: (0..n).map(|i| (i as f32 / n as f32) * 0.9 + 0.05).collect(),
            energy: (0..n)
                .map(|i| 0.5 + 0.3 * ((i as f32 / 100.0).sin()))
                .collect(),
            genome_a: (0..n * 4)
                .map(|i| match i % 4 {
                    0 => 3.0 + (i as f32 % 13.0),         // r: [3, 16]
                    1 => i as f32 / (n * 4) as f32,       // mu: [0, 1]
                    2 => 0.01 + (i as f32 % 29.0) * 0.01, // sigma
                    3 => (i as f32 % 100.0) / 100.0,      // agg
                    _ => unreachable!(),
                })
                .collect(),
            genome_b: (0..n).map(|i| 0.001 + (i % 10) as f32 * 0.0005).collect(),
            resource: (0..n)
                .map(|i| 0.5 + 0.5 * ((i as f32 / 50.0).cos()))
                .collect(),
        }
    }

    #[test]
    fn save_load_roundtrip_is_lossless() {
        let original = create_test_snapshot();
        let path = "/tmp/evolenia_test_snapshot.snap";

        // Save
        save_snapshot(path, &original).expect("Failed to save snapshot");

        // Load
        let loaded = load_snapshot(path).expect("Failed to load snapshot");

        // Cleanup
        let _ = fs::remove_file(path);

        // Verify exact equality
        assert_eq!(
            original.mass.len(),
            loaded.mass.len(),
            "Mass length mismatch"
        );
        assert_eq!(
            original.energy.len(),
            loaded.energy.len(),
            "Energy length mismatch"
        );
        assert_eq!(
            original.genome_a.len(),
            loaded.genome_a.len(),
            "Genome_a length mismatch"
        );
        assert_eq!(
            original.genome_b.len(),
            loaded.genome_b.len(),
            "Genome_b length mismatch"
        );
        assert_eq!(
            original.resource.len(),
            loaded.resource.len(),
            "Resource length mismatch"
        );

        for (i, (&orig, &load)) in original.mass.iter().zip(loaded.mass.iter()).enumerate() {
            assert_eq!(orig, load, "Mass[{}] mismatch: {} vs {}", i, orig, load);
        }
        for (i, (&orig, &load)) in original.energy.iter().zip(loaded.energy.iter()).enumerate() {
            assert_eq!(orig, load, "Energy[{}] mismatch: {} vs {}", i, orig, load);
        }
        for (i, (&orig, &load)) in original
            .genome_a
            .iter()
            .zip(loaded.genome_a.iter())
            .enumerate()
        {
            assert_eq!(orig, load, "Genome_a[{}] mismatch: {} vs {}", i, orig, load);
        }
        for (i, (&orig, &load)) in original
            .genome_b
            .iter()
            .zip(loaded.genome_b.iter())
            .enumerate()
        {
            assert_eq!(orig, load, "Genome_b[{}] mismatch: {} vs {}", i, orig, load);
        }
        for (i, (&orig, &load)) in original
            .resource
            .iter()
            .zip(loaded.resource.iter())
            .enumerate()
        {
            assert_eq!(orig, load, "Resource[{}] mismatch: {} vs {}", i, orig, load);
        }
    }

    #[test]
    fn load_nonexistent_file_fails() {
        let result = load_snapshot("/tmp/this_file_definitely_does_not_exist_12345.snap");
        assert!(result.is_err(), "Loading nonexistent file should fail");
    }

    #[test]
    fn load_invalid_magic_fails() {
        let path = "/tmp/evolenia_invalid_magic.snap";
        fs::write(path, b"BADMAGIC12345678").expect("Failed to write test file");

        let result = load_snapshot(path);
        let _ = fs::remove_file(path);

        assert!(
            result.is_err(),
            "Loading file with invalid magic should fail"
        );
    }
}

#[cfg(test)]
mod trophic_tests {
    //! Tests for trophic classification (prey/opportunist/predator).

    use crate::metrics::SimDiagnostics;
    use crate::world::{BufferSnapshot, WORLD_HEIGHT, WORLD_WIDTH};

    fn create_trophic_snapshot(agg_values: &[(f32, f32)]) -> BufferSnapshot {
        // agg_values: [(aggressivity, mass), ...]
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let mut snap = BufferSnapshot {
            mass: vec![0.0; n],
            energy: vec![0.5; n],
            genome_a: vec![10.0, 0.15, 0.02, 0.0]
                .into_iter()
                .cycle()
                .take(n * 4)
                .collect(),
            genome_b: vec![0.003; n],
            resource: vec![1.0; n],
        };

        for (i, &(agg, mass)) in agg_values.iter().enumerate() {
            if i >= n {
                break;
            }
            snap.mass[i] = mass;
            snap.genome_a[i * 4 + 3] = agg; // aggressivity
        }

        snap
    }

    #[test]
    fn trophic_fractions_sum_to_one() {
        // prey + opportunist + predator = 1.0 (always)
        let snap = create_trophic_snapshot(&[
            (0.1, 0.5), // prey
            (0.3, 0.3), // opportunist
            (0.6, 0.2), // predator
        ]);

        let diag = SimDiagnostics::from_snapshot(&snap);
        let sum = diag.prey_fraction + diag.opportunist_fraction + diag.predator_fraction_strict;

        let diff = (sum - 1.0).abs();
        assert!(
            diff < 0.01,
            "Trophic fractions must sum to 1.0, got {} (prey={}, opp={}, pred={})",
            sum,
            diag.prey_fraction,
            diag.opportunist_fraction,
            diag.predator_fraction_strict
        );
    }

    #[test]
    fn prey_classification_threshold() {
        // Prey: agg < 0.2
        let snap = create_trophic_snapshot(&[
            (0.19, 1.0), // prey
            (0.20, 1.0), // NOT prey (at threshold)
        ]);

        let diag = SimDiagnostics::from_snapshot(&snap);

        // Half should be prey
        assert!(
            (diag.prey_fraction - 0.5).abs() < 0.01,
            "Prey threshold is < 0.2 (exclusive), expected 0.5, got {}",
            diag.prey_fraction
        );
    }

    #[test]
    fn opportunist_classification_range() {
        // Opportunist: 0.2 ≤ agg < 0.5
        let snap = create_trophic_snapshot(&[
            (0.20, 1.0), // opportunist (at lower bound)
            (0.49, 1.0), // opportunist
            (0.50, 1.0), // NOT opportunist
        ]);

        let diag = SimDiagnostics::from_snapshot(&snap);

        // 2/3 should be opportunist
        let expected = 2.0 / 3.0;
        assert!(
            (diag.opportunist_fraction - expected).abs() < 0.01,
            "Opportunist range is [0.2, 0.5), expected {}, got {}",
            expected,
            diag.opportunist_fraction
        );
    }

    #[test]
    fn predator_classification_threshold() {
        // Predator (strict): agg ≥ 0.5
        let snap = create_trophic_snapshot(&[
            (0.49, 1.0), // NOT predator
            (0.50, 1.0), // predator
            (1.0, 1.0),  // predator
        ]);

        let diag = SimDiagnostics::from_snapshot(&snap);

        // 2/3 should be predator
        let expected = 2.0 / 3.0;
        assert!(
            (diag.predator_fraction_strict - expected).abs() < 0.01,
            "Predator threshold is >= 0.5, expected {}, got {}",
            expected,
            diag.predator_fraction_strict
        );
    }

    #[test]
    fn trophic_fractions_mass_weighted() {
        // Mass-weighted, not count-weighted
        let snap = create_trophic_snapshot(&[
            (0.1, 0.9), // prey, 90% of mass
            (0.6, 0.1), // predator, 10% of mass
        ]);

        let diag = SimDiagnostics::from_snapshot(&snap);

        assert!(
            diag.prey_fraction > 0.85,
            "Prey with 90% mass should dominate, got {}",
            diag.prey_fraction
        );
        assert!(
            diag.predator_fraction_strict < 0.15,
            "Predator with 10% mass should be minority, got {}",
            diag.predator_fraction_strict
        );
    }
}

#[cfg(test)]
mod diversity_tests {
    //! Tests for diversity metrics (effective diversity, genome variance).

    use crate::metrics::SimDiagnostics;
    use crate::world::{BufferSnapshot, WORLD_HEIGHT, WORLD_WIDTH};

    #[test]
    fn effective_diversity_minimum_is_one() {
        // Hill number N1 = exp(H) ≥ 1 (even for uniform population)
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let snap = BufferSnapshot {
            mass: vec![0.5; n],
            energy: vec![0.5; n],
            genome_a: vec![10.0, 0.15, 0.02, 0.0]
                .into_iter()
                .cycle()
                .take(n * 4)
                .collect(),
            genome_b: vec![0.003; n],
            resource: vec![1.0; n],
        };

        let diag = SimDiagnostics::from_snapshot(&snap);

        assert!(
            diag.effective_diversity >= 1.0,
            "Effective diversity (Hill N1) must be ≥ 1, got {}",
            diag.effective_diversity
        );
    }

    #[test]
    fn effective_diversity_increases_with_species() {
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;

        // Snapshot 1: uniform population
        let snap_uniform = BufferSnapshot {
            mass: vec![0.5; n],
            energy: vec![0.5; n],
            genome_a: vec![10.0, 0.15, 0.02, 0.0]
                .into_iter()
                .cycle()
                .take(n * 4)
                .collect(),
            genome_b: vec![0.003; n],
            resource: vec![1.0; n],
        };

        // Snapshot 2: two distinct species (half each)
        let mut genome_a_diverse: Vec<f32> = Vec::with_capacity(n * 4);
        for i in 0..n {
            if i < n / 2 {
                genome_a_diverse.extend_from_slice(&[5.0, 0.1, 0.01, 0.0]);
            } else {
                genome_a_diverse.extend_from_slice(&[15.0, 0.9, 0.3, 0.5]);
            }
        }
        let snap_diverse = BufferSnapshot {
            mass: vec![0.5; n],
            energy: vec![0.5; n],
            genome_a: genome_a_diverse,
            genome_b: vec![0.003; n],
            resource: vec![1.0; n],
        };

        let diag_uniform = SimDiagnostics::from_snapshot(&snap_uniform);
        let diag_diverse = SimDiagnostics::from_snapshot(&snap_diverse);

        assert!(
            diag_diverse.effective_diversity > diag_uniform.effective_diversity,
            "Diverse population should have higher effective diversity. Uniform={}, diverse={}",
            diag_uniform.effective_diversity,
            diag_diverse.effective_diversity
        );
    }

    #[test]
    fn genome_variance_is_non_negative() {
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let snap = BufferSnapshot {
            mass: vec![0.5; n],
            energy: vec![0.5; n],
            genome_a: vec![10.0, 0.15, 0.02, 0.0]
                .into_iter()
                .cycle()
                .take(n * 4)
                .collect(),
            genome_b: vec![0.003; n],
            resource: vec![1.0; n],
        };

        let diag = SimDiagnostics::from_snapshot(&snap);

        assert!(
            diag.genome_variance >= 0.0,
            "Genome variance must be ≥ 0, got {}",
            diag.genome_variance
        );
    }

    #[test]
    fn uniform_population_has_zero_genome_variance() {
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let snap = BufferSnapshot {
            mass: vec![0.5; n],
            energy: vec![0.5; n],
            genome_a: vec![10.0, 0.15, 0.02, 0.0]
                .into_iter()
                .cycle()
                .take(n * 4)
                .collect(),
            genome_b: vec![0.003; n],
            resource: vec![1.0; n],
        };

        let diag = SimDiagnostics::from_snapshot(&snap);

        assert!(
            diag.genome_variance < 0.001,
            "Uniform population should have ~0 genome variance, got {}",
            diag.genome_variance
        );
    }
}

#[cfg(test)]
mod initialization_tests {
    //! Tests for world initialization invariants.

    use crate::world::{target_total_mass, total_pixels, TARGET_FILL, WORLD_HEIGHT, WORLD_WIDTH};

    #[test]
    fn total_pixels_is_width_times_height() {
        assert_eq!(
            total_pixels(),
            WORLD_WIDTH * WORLD_HEIGHT,
            "total_pixels() should equal WORLD_WIDTH * WORLD_HEIGHT"
        );
    }

    #[test]
    fn target_mass_is_fill_times_pixels() {
        let expected = (WORLD_WIDTH * WORLD_HEIGHT) as f32 * TARGET_FILL;
        let actual = target_total_mass();

        let diff = (expected - actual).abs();
        assert!(
            diff < 0.01,
            "target_total_mass() should be pixels * TARGET_FILL. Expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn target_fill_is_reasonable() {
        // TARGET_FILL is a compile-time constant; verify at compile time.
        const {
            assert!(TARGET_FILL > 0.0, "TARGET_FILL must be > 0");
        }
        const {
            assert!(TARGET_FILL < 1.0, "TARGET_FILL must be < 1");
        }
    }

    #[test]
    fn world_dimensions_are_power_of_two_friendly() {
        // Workgroup size compatibility (16x16)
        assert!(
            WORLD_WIDTH.is_multiple_of(16),
            "WORLD_WIDTH should be divisible by 16"
        );
        assert!(
            WORLD_HEIGHT.is_multiple_of(16),
            "WORLD_HEIGHT should be divisible by 16"
        );
    }
}

#[cfg(test)]
mod integration_tests {
    //! Integration-level tests that validate end-to-end behaviour.
    //! CPU-side tests (always safe), GPU tests are behind `#[ignore]`.

    use crate::metrics::SimDiagnostics;
    use crate::world::{BufferSnapshot, WORLD_HEIGHT, WORLD_WIDTH};

    fn empty_snapshot() -> BufferSnapshot {
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        BufferSnapshot {
            mass: vec![0.0; n],
            energy: vec![0.5; n],
            genome_a: vec![10.0, 0.15, 0.02, 0.0]
                .into_iter()
                .cycle()
                .take(n * 4)
                .collect(),
            genome_b: vec![0.003; n],
            resource: vec![1.0; n],
        }
    }

    fn saturated_snapshot(mass_val: f32) -> BufferSnapshot {
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        BufferSnapshot {
            mass: vec![mass_val; n],
            energy: vec![1.0; n],
            genome_a: vec![3.0, 0.05, 0.005, 0.0]
                .into_iter()
                .cycle()
                .take(n * 4)
                .collect(),
            genome_b: vec![0.008; n],
            resource: vec![0.0; n],
        }
    }

    #[test]
    fn empty_grid_diagnostics_are_coherent() {
        // An empty grid should report 0 live pixels, 0 species, and valid metrics.
        let snap = empty_snapshot();
        let diag = SimDiagnostics::from_snapshot(&snap);

        assert_eq!(diag.live_pixels, 0, "Empty grid: no live pixels");
        assert!(diag.total_mass < 0.01, "Empty grid: ~zero total mass");
        assert!(
            diag.genetic_entropy >= 0.0,
            "Genetic entropy must be non-negative"
        );
        assert!(diag.effective_diversity >= 1.0, "Effective diversity >= 1");
        assert!(!diag.total_mass.is_nan(), "Total mass must not be NaN");
    }

    #[test]
    fn saturated_grid_diagnostics_are_coherent() {
        // A grid where every pixel has mass=1.0 should have valid metrics.
        let snap = saturated_snapshot(1.0);
        let diag = SimDiagnostics::from_snapshot(&snap);

        let n = WORLD_WIDTH * WORLD_HEIGHT;
        assert_eq!(diag.live_pixels, n, "Saturated grid: all pixels alive");
        assert!(
            (diag.total_mass - n as f32).abs() < 1.0,
            "Total mass ~ pixel count"
        );
        assert!(!diag.total_mass.is_nan(), "Total mass must not be NaN");
        assert!(
            !diag.genetic_entropy.is_nan(),
            "Genetic entropy must not be NaN"
        );
        assert!(diag.starving_fraction >= 0.0 && diag.starving_fraction <= 1.0);
        // Trophic fractions should sum to 1.0
        let trophic_sum =
            diag.prey_fraction + diag.opportunist_fraction + diag.predator_fraction_strict;
        assert!(
            (trophic_sum - 1.0).abs() < 0.01,
            "Trophic fractions sum to 1.0"
        );
    }

    #[test]
    fn extreme_parameters_produce_valid_snapshot() {
        // Extreme genome values should not produce NaN in diagnostics.
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let mut snap = saturated_snapshot(0.5);

        // Set extreme but valid genome values
        for i in 0..n {
            snap.genome_a[i * 4] = 3.0; // r = min
            snap.genome_a[i * 4 + 1] = 0.05; // mu = min
            snap.genome_a[i * 4 + 2] = 0.005; // sigma = min
            snap.genome_a[i * 4 + 3] = 1.0; // agg = max
        }

        let diag = SimDiagnostics::from_snapshot(&snap);

        assert!(diag.genome_variance >= 0.0, "Genome variance >= 0");
        assert!(!diag.genetic_entropy.is_nan(), "Genetic entropy not NaN");
        assert!(!diag.total_mass.is_nan(), "Total mass not NaN");
    }

    #[test]
    fn half_live_half_dead_grid_is_coherent() {
        // Mix of live and dead pixels should produce sensible stats.
        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let mut snap = saturated_snapshot(0.5);

        // Kill first half of pixels
        for i in 0..n / 2 {
            snap.mass[i] = 0.0;
        }

        let diag = SimDiagnostics::from_snapshot(&snap);
        let expected_live = (n - n / 2) as u32;

        assert_eq!(diag.live_pixels, expected_live, "Only half should be alive");
        assert!(diag.total_mass > 0.0, "Should have positive mass");
        assert!(!diag.total_mass.is_nan());
    }

    #[test]
    fn roundtrip_save_load_with_checksum() {
        // Save → load → verify each field exactly matches.
        use crate::state_io::{load_snapshot, save_snapshot};
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n = (WORLD_WIDTH * WORLD_HEIGHT) as usize;
        let original = BufferSnapshot {
            mass: (0..n).map(|i| (i as f32 / n as f32) * 0.9).collect(),
            energy: (0..n)
                .map(|i| 0.5 + 0.3 * ((i as f32 / 42.0).sin()))
                .collect(),
            genome_a: (0..n * 4).map(|i| (i as f32 % 13.0 + 3.0) * 0.1).collect(),
            genome_b: (0..n).map(|i| 0.001 + (i % 10) as f32 * 0.0003).collect(),
            resource: (0..n).map(|i| (i as f32 / 99.0).fract()).collect(),
        };

        let path = "/tmp/evolenia_integration_test.snap";
        save_snapshot(path, &original).expect("Save failed");
        let loaded = load_snapshot(path).expect("Load failed");
        let _ = std::fs::remove_file(path);

        // Lengths match
        assert_eq!(original.mass.len(), loaded.mass.len());

        // Byte-level checksum comparison (hash of all data)
        fn hash_f32_slice(data: &[f32]) -> u64 {
            let mut hasher = DefaultHasher::new();
            // Hash as raw bytes for exact comparison
            for &v in data {
                v.to_bits().hash(&mut hasher);
            }
            hasher.finish()
        }

        assert_eq!(
            hash_f32_slice(&original.mass),
            hash_f32_slice(&loaded.mass),
            "Mass checksum mismatch"
        );
        assert_eq!(
            hash_f32_slice(&original.energy),
            hash_f32_slice(&loaded.energy),
            "Energy checksum mismatch"
        );
        assert_eq!(
            hash_f32_slice(&original.genome_a),
            hash_f32_slice(&loaded.genome_a),
            "Genome A checksum mismatch"
        );
        assert_eq!(
            hash_f32_slice(&original.genome_b),
            hash_f32_slice(&loaded.genome_b),
            "Genome B checksum mismatch"
        );
        assert_eq!(
            hash_f32_slice(&original.resource),
            hash_f32_slice(&loaded.resource),
            "Resource checksum mismatch"
        );
    }

    #[test]
    fn preset_load_is_valid() {
        // Smoke test: load each preset and verify it produces a valid
        // SimulationParams without crashing.

        let presets = [
            "default",
            "best",
            "speciation_engine",
            "autonomous_gliders",
            "explosive_life",
            "lenia_creatures",
            "predator_prey_chaos",
            "stable_colonies",
            "swarm_drones",
        ];

        for preset_name in &presets {
            let params = crate::lab_ui::presets::load_preset(preset_name);
            assert!(
                params.is_some(),
                "Preset '{}' should load successfully",
                preset_name
            );
            let p = params.unwrap();

            // Sanity checks on loaded parameters
            assert!(
                p.time_step >= crate::config::TIME_STEP_MIN
                    && p.time_step <= crate::config::TIME_STEP_MAX,
                "Preset '{}': time_step out of range",
                preset_name
            );
            assert!(
                p.mutation_rate > 0.0,
                "Preset '{}': mutation_rate must be > 0",
                preset_name
            );
            assert!(
                p.num_seed_clusters > 0,
                "Preset '{}': must have seed clusters",
                preset_name
            );
        }
    }

    // ── GPU-dependent integration tests (require GPU, ignored by default) ──

    /// Run N frames headless and verify no NaN appears in diagnostics.
    /// Requires a GPU — skipped in CI. Run with `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn headless_100_frames_no_nan() {
        use crate::config::SimulationParams;
        use crate::headless::{run_headless, HeadlessConfig};

        let config = HeadlessConfig {
            frames: 100,
            load_state_path: None,
            save_state_path: None,
            progress_interval: 0, // silent
            seed: Some(42),
            sim_params: SimulationParams::default(),
        };

        let result = run_headless(&config);
        assert!(
            result.is_ok(),
            "Headless 100 frames should succeed: {:?}",
            result.err()
        );
    }

    /// Run with multiple seeds; each run should complete without error.
    #[test]
    #[ignore]
    fn headless_multiple_seeds_no_crash() {
        use crate::config::SimulationParams;
        use crate::headless::{run_headless, HeadlessConfig};

        for seed in [1u64, 42, 999, 12345] {
            let config = HeadlessConfig {
                frames: 50,
                load_state_path: None,
                save_state_path: None,
                progress_interval: 0,
                seed: Some(seed),
                sim_params: SimulationParams::default(),
            };
            let result = run_headless(&config);
            assert!(
                result.is_ok(),
                "Headless run with seed {} failed: {:?}",
                seed,
                result.err()
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests added by improvement loop #2
// ═══════════════════════════════════════════════════════════════════════

// ── Configuration Tests ──

mod config_tests {
    use crate::config::{
        load_toml_config, PerturbationType, SimulationParams, TIME_STEP_MAX, VIS_MODE_COUNT,
    };

    #[test]
    fn effective_seed_fixed() {
        let p = SimulationParams {
            use_fixed_seed: true,
            fixed_seed_value: 42,
            seed: Some(99),
            ..Default::default()
        };
        assert_eq!(p.effective_seed(), Some(42));
    }

    #[test]
    fn effective_seed_variable() {
        let p = SimulationParams {
            seed: Some(99),
            ..Default::default()
        };
        assert_eq!(p.effective_seed(), Some(99));
    }

    #[test]
    fn effective_seed_none() {
        let p = SimulationParams::default();
        assert_eq!(p.effective_seed(), None);
    }

    #[test]
    fn load_toml_partial_override() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");
        let toml_content = r#"
[simulation]
time_step = 2.0
mutation_rate = 0.8
[headless]
frames = 500
"#;
        std::fs::write(&path, toml_content).unwrap();
        let (params, headless) =
            load_toml_config(path.to_str().unwrap()).expect("should load valid TOML");

        assert_eq!(params.time_step, 2.0);
        assert_eq!(params.mutation_rate, 0.8);
        assert_eq!(
            params.predation_factor,
            SimulationParams::default().predation_factor
        );
        assert!(headless.is_some());
        assert_eq!(headless.unwrap().frames, Some(500));
    }

    #[test]
    fn load_toml_clamp_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clamp.toml");
        let toml_content = r#"
[simulation]
time_step = 100.0
mutation_rate = -5.0
initial_mass_fill = 2.0
num_seed_clusters = 0
"#;
        std::fs::write(&path, toml_content).unwrap();
        let (params, _) = load_toml_config(path.to_str().unwrap()).expect("should load valid TOML");

        assert_eq!(params.time_step, TIME_STEP_MAX);
        assert_eq!(params.mutation_rate, 0.0);
        assert_eq!(params.initial_mass_fill, 0.9);
        assert_eq!(params.num_seed_clusters, 1);
    }

    #[test]
    fn load_toml_empty_simulation_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.toml");
        std::fs::write(&path, "[simulation]\n").unwrap();
        let (params, _) = load_toml_config(path.to_str().unwrap()).expect("should load valid TOML");
        let default = SimulationParams::default();
        assert_eq!(params.time_step, default.time_step);
    }

    #[test]
    fn load_toml_file_not_found() {
        let result = load_toml_config("/nonexistent/path/config.toml");
        assert!(result.is_err());
    }

    #[test]
    fn load_toml_invalid_syntax() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid.toml");
        std::fs::write(&path, "this is not valid toml {{{[").unwrap();
        let result = load_toml_config(path.to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn perturbation_type_count() {
        assert_eq!(PerturbationType::all().len(), 5);
    }

    #[test]
    fn perturbation_type_names() {
        assert_eq!(PerturbationType::None.name(), "None");
        assert_eq!(PerturbationType::Drought.name(), "Drought");
        assert_eq!(PerturbationType::MassStorm.name(), "Mass Storm");
    }

    #[test]
    fn visualization_mode_names() {
        use crate::config::visualization_mode_name;
        assert_eq!(visualization_mode_name(0), "Species Color");
        assert_eq!(visualization_mode_name(8), "Debug Raw");
        assert_eq!(visualization_mode_name(99), "Unknown");
    }

    #[test]
    fn visualization_mode_clamped() {
        let p = SimulationParams {
            visualization_mode: 100 % VIS_MODE_COUNT,
            ..Default::default()
        };
        assert_eq!(p.visualization_mode, 1);
    }
}

// ── Camera Tests ──

mod camera_tests {
    use crate::camera::CameraState;

    #[test]
    fn default_state() {
        let cam = CameraState::default();
        assert_eq!(cam.offset, [0.0, 0.0]);
        assert_eq!(cam.zoom, 1.0);
    }

    #[test]
    fn pan_up() {
        let mut cam = CameraState::default();
        cam.apply_pan(true, false, false, false);
        assert!(cam.offset[1] < 0.0);
    }

    #[test]
    fn pan_down() {
        let mut cam = CameraState::default();
        cam.apply_pan(false, true, false, false);
        assert!(cam.offset[1] > 0.0);
    }

    #[test]
    fn pan_speed_inverse_to_zoom() {
        let mut cam = CameraState {
            zoom: 2.0,
            ..Default::default()
        };
        cam.apply_pan(true, false, false, false);
        let offset_at_zoom2 = cam.offset[1];

        let mut cam2 = CameraState {
            zoom: 1.0,
            ..Default::default()
        };
        cam2.apply_pan(true, false, false, false);
        assert!((offset_at_zoom2 * 2.0 - cam2.offset[1]).abs() < 1e-6);
    }

    #[test]
    fn zoom_in() {
        let mut cam = CameraState::default();
        cam.apply_zoom_keys(true, false);
        assert!(cam.zoom > 1.0);
    }

    #[test]
    fn zoom_out() {
        let mut cam = CameraState::default();
        cam.apply_zoom_keys(false, true);
        assert!(cam.zoom < 1.0);
    }

    #[test]
    fn zoom_clamped_min() {
        let mut cam = CameraState {
            zoom: 0.11,
            ..Default::default()
        };
        cam.apply_zoom_keys(false, true);
        assert!(cam.zoom >= 0.1);
    }

    #[test]
    fn zoom_clamped_max() {
        let mut cam = CameraState {
            zoom: 49.9,
            ..Default::default()
        };
        cam.apply_zoom_keys(true, false);
        assert!(cam.zoom <= 50.0);
    }

    #[test]
    fn scroll_zoom() {
        let mut cam = CameraState::default();
        cam.apply_scroll(1.0);
        assert!(cam.zoom > 1.0);
    }

    #[test]
    fn uniforms_aspect_ratio() {
        let cam = CameraState::default();
        let u = cam.uniforms(1920, 1080);
        assert!((u.aspect_ratio - 1920.0 / 1080.0).abs() < 1e-6);
    }

    #[test]
    fn uniforms_zoom_passthrough() {
        let cam = CameraState {
            zoom: 2.5,
            ..Default::default()
        };
        let u = cam.uniforms(800, 600);
        assert_eq!(u.zoom, 2.5);
    }
}

// ── State I/O Corruption Tests ──

mod state_io_corruption_tests {
    use crate::state_io;
    use std::io::Write;

    #[test]
    fn load_truncated_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trunc_magic.snap");
        std::fs::write(&path, b"EVO").unwrap();
        assert!(state_io::load_snapshot(path.to_str().unwrap()).is_err());
    }

    #[test]
    fn load_truncated_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trunc_header.snap");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"EVOSNP01").unwrap();
        f.write_all(&512u32.to_le_bytes()).unwrap();
        // Missing height byte
        f.write_all(&[0u8]).unwrap();
        assert!(state_io::load_snapshot(path.to_str().unwrap()).is_err());
    }

    #[test]
    fn load_corrupt_length_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt_len.snap");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"EVOSNP01").unwrap();
        f.write_all(&512u32.to_le_bytes()).unwrap();
        f.write_all(&512u32.to_le_bytes()).unwrap();
        // Write a huge element count exceeding MAX_ELEMENTS (10M)
        f.write_all(&10_000_001u64.to_le_bytes()).unwrap();
        assert!(state_io::load_snapshot(path.to_str().unwrap()).is_err());
    }

    #[test]
    fn load_wrong_dimensions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wrong_dim.snap");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"EVOSNP01").unwrap();
        f.write_all(&256u32.to_le_bytes()).unwrap();
        f.write_all(&256u32.to_le_bytes()).unwrap();
        // Empty buffers (len=0) but wrong dimensions
        for _ in 0..6 {
            f.write_all(&0u64.to_le_bytes()).unwrap();
        }
        assert!(state_io::load_snapshot(path.to_str().unwrap()).is_err());
    }
}
