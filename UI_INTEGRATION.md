# 🎨 Guide d'Intégration UI Interactive - EvoLenia v2

## ✅ Ce qui a été fait

1. **Ajout d'egui** à Cargo.toml
2. **Création du module ui.rs** avec:
   - `SimulationParams` : Tous les paramètres ajustables
   - `SimulationStats` : Statistiques en temps réel  
   - `render_ui()` : Interface interactive complète

3. **Modification d'AppState** pour inclure egui

## 🔧 Intégration Complète (À faire)

### Étape 1 : Modifier window_event pour capturer les événements egui

Dans `fn window_event()`, avant le match sur event :

```rust
// Passer l'événement à egui d'abord
let response = state.egui_state.on_window_event(&state.window, &event);
if response.consumed {
    return; // egui a consommé l'événement
}
```

### Étape 2 : Remplacer les références

Remplacer partout :
- `state.paused` → `state.sim_params.paused`
- `state.visualization_mode` → `VisualizationMode` (géré dynamiquement)

### Étape 3 : Modifier la boucle de rendu (RedrawRequested)

Remplacer la section glyphon par :

```rust
// Préparer l'UI egui
let raw_input = state.egui_state.take_egui_input(&state.window);
let full_output = state.egui_ctx.run(raw_input, |ctx| {
    render_ui(ctx, &mut state.sim_params, &state.sim_stats, &mut state.restart_requested);
});

// Gérer le restart si demandé
if state.restart_requested {
    state.world = WorldState::new_with_params(
        &state.device,
        state.sim_params.num_seed_clusters,
        state.sim_params.seed_cluster_size,
        state.sim_params.initial_mass_fill,
    );
    state.pipelines = create_pipelines(&state.device, &state.world, state.surface_config.format);
    state.restart_requested = false;
    log::info!("Simulation restarted with new parameters");
}

// Mettre à jour les stats
state.sim_stats.frame = state.world.frame;
state.sim_stats.fps = state.fps;
// TODO: calculer les autres stats (total_mass, avg_energy, etc.)
```

Puis dans le render pass, après le rendu de la simulation :

```rust
// Rendu egui par dessus
state.egui_state.handle_platform_output(&state.window, full_output.platform_output);

let screen_descriptor = egui_wgpu::ScreenDescriptor {
    size_in_pixels: [state.surface_config.width, state.surface_config.height],
    pixels_per_point: state.window.scale_factor() as f32,
};

let tris = state.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

for (id, image_delta) in &full_output.textures_delta.set {
    state.egui_renderer.update_texture(&state.device, &state.queue, *id, image_delta);
}

state.egui_renderer.update_buffers(
    &state.device,
    &state.queue,
    &mut encoder,
    &tris,
    &screen_descriptor,
);

{
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("egui_render_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load, // Garder le rendu précédent
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    state.egui_renderer.render(&mut render_pass, &tris, &screen_descriptor);
}

for id in &full_output.textures_delta.free {
    state.egui_renderer.free_texture(id);
}
```

### Étape 4 : Adapter WorldState::new() pour accepter des paramètres

Ajouter dans world.rs :

```rust
impl WorldState {
    pub fn new_with_params(
        device: &wgpu::Device,
        num_clusters: usize,
        cluster_size: f32,
        mass_fill: f32,
    ) -> Self {
        // Créer le monde avec les paramètres personnalisés
        // ...
    }
}
```

## 🎮 Fonctionnalités de l'UI

### Panneau de Contrôle
- ▶️ Pause/Resume
- 🔄 Restart avec nouveaux paramètres
- 🎨 Sélection du mode de visualisation

### Paramètres Ajustables
- **Time Stepping** : DT (0.01 - 0.2)
- **Initial Conditions** :
  - Nombre de clusters (1-200)
  - Taille des clusters (1-10)
  - Remplissage de masse (5%-50%)
- **Évolution** :
  - Force de mutation (0-5x)
  - Taux de mutation de base (0-2%)
  - Prédation on/off
- **Ressources** :
  - Diffusion (0-0.5)
  - Régénération (0-0.1)
  - Consommation (0-0.2)
- **Conservation** :
  - Normalisation on/off
  - Multiplicateur de masse cible (0.5-2x)

### Statistiques en Temps Réel
- 🎬 Frame actuel
- ⚡ FPS
- ⚖️ Masse totale
- 🔋 Énergie moyenne
- 🟢 Pixels vivants
- 🔴 Nombre de prédateurs

### Raccourcis Clavier (panel d'aide F1)
- Tous les contrôles existants + F1 pour basculer l'aide

## 🚀 Utilisation Après Intégration

Lancer normalement :
```bash
cargo run --release
```

L'UI apparaît automatiquement avec tous les contrôles interactifs !

## 📝 TODO

- [ ] Finir l'intégration dans window_event
- [ ] Ajouter le rendu egui dans la boucle
- [ ] Implémenter new_with_params dans world.rs
- [ ] Calculer les statistiques en temps réel
- [ ] Tester et débugger

## 💡 Avantages

✅ **Contrôle total** sur tous les paramètres sans recompiler  
✅ **Feedback visuel** immédiat avec les sliders  
✅ **Expérimentation facile** pour trouver les meilleurs paramètres  
✅ **UI professionnelle** avec egui  
✅ **Performances** : egui est très léger (<1% overhead)

---

**Note** : L'intégration est partiellement faite. Il reste à connecter tous les événements et le rendu egui dans la boucle principale. Une fois terminé, vous aurez une UI complète et interactive pour contrôler EvoLenia en temps réel !
