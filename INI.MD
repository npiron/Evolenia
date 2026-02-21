# 📑 DOSSIER DE RECHERCHE & DÉVELOPPEMENT : PROJECT "EVOLENIA" v2.0
> *Révision scientifique complète — Corrections des lacunes formelles de la v1.0*

---

## 1. Vision et Objectifs Scientifiques

**EvoLenia** est un écosystème de vie artificielle (ALife) où l'évolution est **endogène** (interne au système), implémenté en Rust natif avec WGPU.

- **But** : Observer l'émergence spontanée de taxinomies (espèces) numériques **mesurables**.
- **Innovation** : Contrairement à Lenia classique (1 règle globale), chaque pixel possède son propre **génome dynamique à 5 canaux** codé dans une texture GPU, avec un opérateur d'advection garantissant la **conservation de masse**.
- **Moteur** : Physique continue (Fluide + Conservation) + Génétique discrète (Mutations).

### 1.1 Positionnement par rapport à l'état de l'art

| Feature                      | Lenia (Chan 2019) | Flow-Lenia (2025) | **EvoLenia v2 (ce projet)** |
|------------------------------|:-----------------:|:-----------------:|:---------------------------:|
| Règles globales               | ✅                | ✅                | ✅                          |
| Génome spatialisé             | ❌                | ✅                | ✅                          |
| **Conservation de masse**     | ❌                | ✅                | ✅ (corrigé)                |
| **Advection ADN (ségrégée)**  | N/A               | Partielle         | ✅ (novateur)               |
| Mutations endogènes           | ❌                | ✅                | ✅                          |
| **Prédation explicite**       | ❌                | ❌                | ✅ (novateur)               |
| GPU natif (WGPU/Rust)         | ❌                | ❌ (JAX/Python)   | ✅ (avantage compétitif)    |
| **Métriques d'émergence**     | ❌                | Partielle         | ✅ (corrigé)                |

> **Référence principale** : *Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Lenia*, Artificial Life, MIT Press, 2025.

---

## 2. Stack Technique

| Composant        | Technologie                     | Justification                                                          |
|------------------|---------------------------------|------------------------------------------------------------------------|
| Langage          | **Rust** (edition 2021)         | Sûreté mémoire sans GC → déterminisme des simulations                  |
| Backend Calcul   | **WGPU v22+** (WGSL)            | Multi-API (Vulkan, Metal, DX12, WebGPU) → reproductibilité             |
| Architecture     | **Data-Oriented Design**        | Cache-friendly → throughput maximal pour des millions de pixels-agents |
| Parallélisme     | **SIMD massif GPU**             | Chaque pixel = 1 thread indépendant                                    |
| Buffering        | **Ping-Pong Buffers (×2)**      | Évite les race conditions → résultats déterministes                    |
| Fenêtrage        | **winit**                       | Cross-platform window management                                       |
| Métriques        | **Réductions GPU (atomics)**    | Calcul en-place de la diversité génomique sans transfert CPU           |

---

## 3. Les Lois de l'Univers EvoLenia (Formalisme Mathématique Complet)

### 3.1 La Pentade des Champs (Textures 2D)

> ⚠️ **Correction v2** : Le génome passe de 4 à **5 canaux** (`vec4` + buffer séparé ou packing). Les canaux `mu` et `sigma` sont **obligatoires** pour la fonction de croissance de Lenia.

| Champ   | Type           | Description                                         | Plage    |
|---------|----------------|-----------------------------------------------------|----------|
| `M`     | `f32`          | **Masse** : Densité de "matière" de la créature     | [0, 1]   |
| `E`     | `f32`          | **Énergie** : Réserve métabolique                   | [0, 1]   |
| `G.r`   | `f32` (Genome) | **Rayon de perception** `r` (morphologie)           | [2, 16]  |
| `G.g`   | `f32` (Genome) | **Centre de croissance** `μ` (niche écologique)     | [0, 1]   |
| `G.b`   | `f32` (Genome) | **Largeur de croissance** `σ` (tolérance)           | [0.01, 0.3] |
| `G.a`   | `f32` (Genome) | **Agressivité** (prédation)                         | [0, 1]   |
| `G2.r`  | `f32` (Genome) | **Taux de mutation** (instabilité génétique)        | [0, 0.1] |

> Les 5 gènes sont stockés dans **deux textures RGBA** : `genome_a` (`r, μ, σ, agressivité`) et `genome_b.r` (`mutation_rate`).

### 3.2 Formalisme Mathématique de Lenia (Complet)

La règle de Lenia est définie par :

```
U(x, t) = (K_r * M)(x, t)          // Convolution avec noyau K de rayon r
G(u; μ, σ) = exp(-((u - μ)² / (2σ²)))  // Fonction de croissance (cloche gaussienne)
ΔM(x, t)  = 2·G(U(x,t); μ(x), σ(x)) - 1  // Incrément de masse ∈ [-1, +1]
M(x, t+dt) = clamp(M(x,t) + dt · ΔM(x,t), 0, 1)
```

**Le noyau normalisé** (ring kernel de Lenia) :

```
K_r(x) = (1/Z) · exp(-((|x|/r - 0.5)² / (2·0.15²)))
Z = Σ_x K_r(x)    // Facteur de normalisation (somme = 1)
```

> ⚠️ **Correction v2** : Le noyau DOIT être normalisé (`Z`). Sans normalisation, `U` sort de [0,1] et la fonction de croissance diverge.

### 3.3 Conservation de la Masse (Opérateur d'Advection)

> ⚠️ **Correction critique v2** : La masse doit être **transférée**, jamais copiée.

L'advection est calculée via un **champ de vitesse local** `v(x,t)` dérivé du gradient de masse :

```
v(x, t) = ∇M(x, t) · aggressivity(x)  // Prédateurs "remontent" le gradient

// Conservation : flux entrant = flux sortant
M_new(x) = M(x) + Σ_{n ∈ voisins} flux(n → x) - Σ_{n ∈ voisins} flux(x → n)

// flux(x → n) = clamp(v(x,t) · dir(x→n), 0, M(x)/4)
// Limiteur : on ne peut pas transférer plus de M(x)/4 vers un voisin
```

**Vérification de conservation** (à chaque step sur le CPU, en debug) :

```
assert(|Σ_x M_new(x) - Σ_x M(x)| < ε)  // ε = 1e-4
```

### 3.4 Le Métabolisme Corrigé

```
cost(x) = length(vec3(μ, σ, aggressivity)) · M(x) · 0.01
         // Pénalise la complexité génomique (parcimonie de Darwin)

absorption(x) = ResourceMap(x) · M(x) · 0.05
               // ResourceMap : nutriments locaux (dynamique, voir §3.6)

E_new(x) = clamp(E(x) + absorption(x) - cost(x), 0, 1)
```

### 3.5 La Prédation (Mécanisme Corrigé)

> ⚠️ **Correction v2** : La prédation opère via le **champ de flux**, pas par transfert direct.

```
// Un prédateur (G.a élevé) oriente son flux vers les voisins à M faible
predation_flux(x → n) = G.a(x) · clamp(M(x) - M(n), 0, ∞) · 0.02
                       // Ne vole que si M(x) > M(n) (force du plus fort)

// Coût métabolique de la prédation
E(x) -= predation_flux_total(x) · 0.1  // La prédation coûte de l'énergie
```

### 3.6 La ResourceMap Dynamique

> ⚠️ **Correction v2** : La ResourceMap doit être **dynamique** pour créer une pression sélective spatiale.

```
// Réaction-Diffusion de type Gray-Scott pour les nutriments
R_new(x) = R(x) + D_R · ∇²R(x) + feed_rate · (1 - R(x)) - R(x) · M(x) · consumption
         // R(x) ∈ [0,1] : concentration de nutriment
         // D_R = 0.1 : coefficient de diffusion
         // feed_rate = 0.04 : taux de régénération
```

### 3.7 Hérédité & Advection ADN (Ségrégation Génomique)

> ⚠️ **Correction critique v2** : L'advection ADN ne doit PAS être une moyenne (convergence génomique). Utiliser la **ségrégation stochastique** :

```
// Lors du transfert de masse flux(x → n) :
// La masse "emporte" son ADN stochastiquement, pas en moyenne

if (flux > threshold) {
    // Le génome du receveur est remplacé avec probabilité proportionnelle au flux
    p_replace = flux / (M(n) + flux)
    if (random() < p_replace) {
        G_new(n) = G(x)  // Le donneur "colonise" le receveur
    }
    // Sinon, G(n) est conservé (résistance génomique)
}
```

### 3.8 Mutations (Modèle Corrigé)

> ⚠️ **Correction v2** : Le taux de mutation `A` doit avoir un prior pour éviter la dérive vers 0 ou 1.

```
// Mutation avec prior bêta (α=0.5, β=5.0) pour favoriser les faibles taux de mutation
noise = hash_prng(pos, frame_id) * 2.0 - 1.0  // ∈ [-1, +1]

G_new.r     = clamp(G.r     + noise * mut_rate * 14.0,  2.0, 16.0)
G_new.mu    = clamp(G.mu    + noise * mut_rate,          0.0,  1.0)
G_new.sigma = clamp(G.sigma + noise * mut_rate * 0.3,   0.01,  0.3)
G_new.agg   = clamp(G.agg   + noise * mut_rate,          0.0,  1.0)

// Mutation du taux de mutation lui-même (méta-évolution), avec prior
G_new.mut_rate = clamp(
    G.mut_rate + noise * 0.001,  // Pas petit pour la méta-mutation
    0.0001,                       // Minimum : quasi-stable
    0.05                          // Maximum : jamais chaotique
)
```

---

## 4. Architecture GPU Complète (WGPU Pipeline)

### 4.1 Buffers

```
WorldState {
    // Ping-Pong : current (read) / next (write)
    mass_current:     StorageBuffer<f32>[WIDTH * HEIGHT],
    mass_next:        StorageBuffer<f32>[WIDTH * HEIGHT],
    energy_current:   StorageBuffer<f32>[WIDTH * HEIGHT],
    energy_next:      StorageBuffer<f32>[WIDTH * HEIGHT],
    genome_a_current: StorageBuffer<vec4<f32>>[WIDTH * HEIGHT],  // r, mu, sigma, agg
    genome_a_next:    StorageBuffer<vec4<f32>>[WIDTH * HEIGHT],
    genome_b_current: StorageBuffer<f32>[WIDTH * HEIGHT],        // mut_rate
    genome_b_next:    StorageBuffer<f32>[WIDTH * HEIGHT],
    resource_map:     StorageBuffer<f32>[WIDTH * HEIGHT],        // Nutriments
    velocity_field:   StorageBuffer<vec2<f32>>[WIDTH * HEIGHT],  // Flux d'advection
}
```

### 4.2 Passes Compute (Ordre Obligatoire)

```
Frame N:
  Pass 1 : compute_velocity.wgsl    → velocity_field (∇M, prédation)
  Pass 2 : compute_evolution.wgsl   → mass_next, energy_next, genome_a_next, genome_b_next
  Pass 3 : compute_resources.wgsl   → resource_map (Gray-Scott)
  Pass 4 : normalize_mass.wgsl      → correction conservation de masse (global reduction)
  Pass 5 : render.wgsl              → framebuffer display
  Swap    : current ↔ next (ping-pong)
```

---

## 5. Métriques d'Émergence (Sans Transfert CPU)

> ⚠️ **Correction v2** : Sans métriques quantitatives, c'est de l'art génératif, pas de la science.

Les métriques suivantes sont calculées **en GPU** via des passes de réduction atomique :

| Métrique                  | Formule GPU                                        | Interprétation                              |
|---------------------------|-----------------------------------------------------|---------------------------------------------|
| **Masse totale**          | `Σ M(x)`                                           | Conservation (doit être constante)          |
| **Diversité génomique**   | `H = -Σ p(g) · log(p(g))` (Shannon sur G.mu/G.r)  | > 0 = coexistence de taxinomies             |
| **Énergie moyenne**       | `<E>`                                              | Santé globale de l'écosystème               |
| **Variance d'agressivité**| `Var(G.a)`                                         | Présence de la dynamique proie-prédateur    |
| **Taille des cliques**    | Clustering spatial de G (r-voisinage uniforme)     | Taille des "colonies" d'espèces             |

---

## 6. Rendu Visuel (Biomorph Rendering)

```wgsl
// Mapping couleur → génotype visible
let species_color = vec3<f32>(
    genome_a.r / 16.0,   // R = rayon → rouge (petites créatures = rouge vif)
    genome_a.g,           // G = mu    → vert  (niche écologique)
    genome_a.b            // B = sigma → bleu  (tolérance/généralisme)
);
let alpha = mass;  // Opacité = densité de masse

// Surbrillance des prédateurs (agressivité > 0.7 → halo orange)
let predator_glow = step(0.7, genome_a.a) * vec3<f32>(1.0, 0.5, 0.0);
let final_color = species_color + predator_glow * 0.3;
```

---

## 7. 🚀 MASTER PROMPT CORRIGÉ : "EVOLENIA RUST ENGINE v2"

```
Role: You are a World-Class Systems Architect and Artificial Life Researcher,
specializing in Rust, WGPU, and GPU-accelerated simulations.

Project Goal: Implement the "EvoLenia v2" engine — a spatially-varying
continuous cellular automaton with endogenous evolution, mass conservation,
and emergent predation dynamics.

=== 1. Project Structure (Cargo.toml) ===
Dependencies:
  wgpu       = "22"
  winit      = "0.30"
  bytemuck   = { version = "1", features = ["derive"] }
  pollster   = "0.3"
  rand       = "0.8"
  env_logger = "0.11"

=== 2. Constants ===
  WORLD_WIDTH  : u32 = 1024
  WORLD_HEIGHT : u32 = 1024
  WORKGROUP_X  : u32 = 16
  WORKGROUP_Y  : u32 = 16
  DT           : f32 = 0.1

=== 3. Memory Layout (GPU Buffers) ===
Implement a WorldState struct holding, in PING-PONG pairs (current/next):
  - mass         : StorageBuffer<f32>          (1 float per pixel)
  - energy       : StorageBuffer<f32>          (1 float per pixel)
  - genome_a     : StorageBuffer<vec4<f32>>    (r, mu, sigma, aggressivity)
  - genome_b     : StorageBuffer<f32>          (mutation_rate only)
  - resource_map : StorageBuffer<f32>          (single buffer, updated in place)
  - velocity     : StorageBuffer<vec2<f32>>    (advection field, single buffer)
All buffers use STORAGE | COPY_SRC | COPY_DST usage flags.
Use bytemuck for zero-copy casting.

=== 4. Compute Shaders (WGSL) — 4 separate .wgsl files ===

--- 4a. compute_velocity.wgsl ---
  Input  : mass_current, genome_a_current
  Output : velocity (vec2<f32> per pixel)
  Logic  :
    1. Compute gradient ∇M via central differences.
    2. Base velocity = ∇M (mass flows uphill for predators).
    3. Predation component: for each neighbor n,
       if aggressivity > 0.5 AND M(self) > M(n):
         flux_out += aggressivity * (M(self) - M(n)) * 0.02
         velocity oriented toward n
    4. velocity(x) = clamp(aggressivity * ∇M, -1.0, 1.0)

--- 4b. compute_evolution.wgsl ---
  Input  : mass_current, energy_current, genome_a_current, genome_b_current,
           resource_map, velocity
  Output : mass_next, energy_next, genome_a_next, genome_b_next
  Logic  :
  
  [PRNG]
  Implement hash-based PRNG: pcg_hash(state: u32) -> u32
  float_from_hash(h: u32) -> f32 (maps to [-1, +1])
  seed = hash(pixel_index XOR frame_id XOR 0xDEADBEEFu)

  [KERNEL BLENDING]
  Three pre-defined ring kernels for radii 3, 7, 14.
  Interpolate between kernels based on genome_a.r.
  Each kernel cell weight = exp(-((dist/r - 0.5)^2 / (2*0.15^2)))
  Normalize kernel (sum = 1.0) before use.

  [CONVOLUTION & GROWTH]
  U = sum over kernel window of kernel_weight * M(neighbor)
  G(U) = exp(-((U - genome_a.g)^2 / (2 * genome_a.b^2)))
  dM   = 2.0 * G(U) - 1.0
  mass_candidate = clamp(M + DT * dM, 0.0, 1.0)

  [METABOLISM]
  cost        = length(genome_a.gbg) * M * 0.01
  absorption  = resource_map(x) * M * 0.05
  energy_next = clamp(E + absorption - cost, 0.0, 1.0)
  if energy_next <= 0.0: mass_candidate *= 0.99  // entropie douce

  [MASS-CONSERVATIVE ADVECTION]
  For each of 4 cardinal neighbors n:
    flux(self→n) = clamp(velocity · dir(self→n), 0.0, M/4.0)
    flux(n→self) = read from velocity of neighbor n (inward component)
  mass_next = mass_candidate
            + sum(flux_in from neighbors)
            - sum(flux_out to neighbors)
  mass_next = clamp(mass_next, 0.0, 1.0)

  [DNA ADVECTION — STOCHASTIC SEGREGATION]
  For each neighbor n where flux(n→self) > 0.001:
    p_replace = flux(n→self) / (mass_next + 0.001)
    if random() < p_replace:
      genome_a_next = genome_a(n)  // colonisation
      genome_b_next = genome_b(n)
    // Else: génome local conservé

  [MUTATION]
  noise = float_from_hash(seed) * genome_b (mut_rate)
  genome_a_next.r   = clamp(genome_a.r   + noise*14.0, 2.0, 16.0)
  genome_a_next.g   = clamp(genome_a.g   + noise,      0.0,  1.0)
  genome_a_next.b   = clamp(genome_a.b   + noise*0.3,  0.01, 0.3)
  genome_a_next.a   = clamp(genome_a.a   + noise,      0.0,  1.0)
  genome_b_next     = clamp(genome_b     + noise*0.01, 1e-4, 0.05)
  // Appliquer uniquement si mass_next > 0.01 (les cellules mortes ne mutent pas)

--- 4c. compute_resources.wgsl ---
  Input/Output : resource_map (in-place)
  Logic (Gray-Scott simplifié) :
    laplacian_R  = (R(up)+R(down)+R(left)+R(right) - 4*R(x)) / 4.0
    consumed     = R(x) * M(x) * 0.05
    R_new        = clamp(R(x) + 0.1*laplacian_R + 0.04*(1-R(x)) - consumed, 0, 1)

--- 4d. normalize_mass.wgsl ---
  Two-pass reduction:
    Pass A: each workgroup sums its local mass into a StorageBuffer<atomic<u32>>
            using atomicAdd (cast f32 to u32 bits for atomics).
    Pass B: single-thread corrects total:
            correction = TARGET_TOTAL_MASS / actual_total
            mass(x) *= correction  [clamp to [0,1]]
  TARGET_TOTAL_MASS = WORLD_WIDTH * WORLD_HEIGHT * 0.15  // 15% de remplissage initial

=== 5. Rendering Pipeline ===
Vertex shader  : full-screen quad (2 triangles, no vertex buffer).
Fragment shader:
  species_color = vec3(genome_a.r/16, genome_a.g, genome_a.b)
  predator_glow = step(0.7, genome_a.a) * vec3(1.0, 0.5, 0.0)
  final_color   = species_color + predator_glow * 0.3
  output_alpha  = mass
  // Utiliser la texture de masse et de génome comme uniform textures (sampler2D)

=== 6. Initialization (CPU side, main.rs) ===
  Seed random clusters of 20–40 pixels (not uniform noise):
    - Random position, random radius 5–15
    - Genome values drawn from uniform distributions within valid ranges
    - Mass = gaussian falloff from center
    - Energy = 0.5 (uniform)
    - Resource map = 1.0 (full nutrients at start)
  Use at least 50 independent seed clusters.

=== 7. Metrics Collection (optional, CPU readback every 300 frames) ===
  - Copy mass_current to staging buffer
  - Compute on CPU: total_mass, mean_energy
  - Log to stderr as CSV: frame, total_mass, mean_energy
  - Allows verifying mass conservation over time

=== 8. Main Loop (main.rs) ===
  pollster::block_on(async {
    // init wgpu: instance, adapter, device, queue
    // init winit event loop
    // init WorldState and all pipelines
    // event loop:
    //   on RedrawRequested:
    //     run_compute_pass_1(velocity)
    //     run_compute_pass_2(evolution)
    //     run_compute_pass_3(resources)
    //     run_compute_pass_4(normalize)
    //     run_render_pass()
    //     swap ping-pong buffers
    //   on KeyboardInput(Escape): exit
  })

=== Deliverables ===
Provide:
  - Cargo.toml
  - src/main.rs          (WGPU init, pipeline creation, event loop)
  - src/world.rs         (WorldState struct, buffer management, swap)
  - src/shaders/compute_velocity.wgsl
  - src/shaders/compute_evolution.wgsl
  - src/shaders/compute_resources.wgsl
  - src/shaders/normalize_mass.wgsl
  - src/shaders/render.wgsl

=== Quality Requirements ===
  - All WGSL code must be valid WGSL 2024 (no GLSL-isms).
  - All kernel accesses must use wrapping modular indexing (toroidal world).
  - All random number generation must be hash-based (no global state).
  - All biological principles must be commented in the shader code.
  - Mass conservation must hold to within 0.01% per frame.
  - Code must compile with `cargo build --release` without warnings.
```

---

## 8. Critères de Validation Scientifique

Pour que le projet soit considéré comme une **démonstration scientifique** (et non artistique), les phénomènes suivants doivent être observables et mesurables :

| Phénomène                  | Indicateur Mesurable                                      | Seuil de validation         |
|----------------------------|-----------------------------------------------------------|-----------------------------|
| Coexistence d'espèces      | Entropie de Shannon sur G.mu > 1.5 bits                  | Après 10 000 frames         |
| Sélection naturelle        | Corrélation négative `coût ~ survie`                      | p < 0.05 sur 100 runs       |
| Conservation de masse      | `|Σ M_t - Σ M_0| / Σ M_0 < 0.01%`                       | À chaque frame              |
| Dynamique proie-prédateur  | Oscillations de Lotka-Volterra dans `Var(G.a)` vs `<M>`  | Période observable          |
| Diversification génomique  | Augmentation de `Var(G)` dans le temps                   | Monotone sur 5000 frames    |

---

## 9. Glossaire Bio-Mathématique

| Terme                 | Définition dans EvoLenia                                                      |
|-----------------------|-------------------------------------------------------------------------------|
| **Noyau `K_r`**       | Ring kernel de Lenia : détecte la densité locale à une distance ≈ r/2        |
| **`U(x,t)`**          | Signal de densité perçu par la cellule (résultat de la convolution)          |
| **`G(U; μ, σ)`**      | Fonction de croissance en cloche : favorise les densités autour de μ         |
| **`ΔM`**              | Incrément de masse ∈ [-1,+1] : positif = croissance, négatif = dissipation   |
| **Advection ADN**     | Transport du génome par le flux de masse (hérédité spatiale)                 |
| **Ségrégation**       | Décision stochastique : le génome d'une cellule "gagne" ou "résiste"         |
| **Prior bêta sur A**  | Distribution de probabilité qui évite que le taux de mutation dérive à 0/1   |
| **Gray-Scott**        | Système de réaction-diffusion modélisant la dynamique des nutriments         |
| **Conservation**      | Propriété fondamentale : `Σ M` reste constant (ni création ni destruction)   |