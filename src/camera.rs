// ============================================================================
// camera.rs — EvoLenia v2
// Camera state & GPU uniform for pan/zoom navigation.
// ============================================================================

/// GPU-side camera uniforms uploaded every frame.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub offset: [f32; 2],
    pub zoom: f32,
    pub _pad: f32,
}

impl Default for CameraUniforms {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            zoom: 1.0,
            _pad: 0.0,
        }
    }
}

/// CPU-side camera state used to track pan/zoom between frames.
pub struct CameraState {
    pub offset: [f32; 2],
    pub zoom: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            zoom: 1.0,
        }
    }
}

impl CameraState {
    /// Apply continuous pan from held keys. Speed is inversely proportional to
    /// zoom so camera movement feels consistent on screen.
    pub fn apply_pan(&mut self, up: bool, down: bool, left: bool, right: bool) {
        let pan_speed = 0.005 / self.zoom;
        if up {
            self.offset[1] -= pan_speed;
        }
        if down {
            self.offset[1] += pan_speed;
        }
        if left {
            self.offset[0] -= pan_speed;
        }
        if right {
            self.offset[0] += pan_speed;
        }
    }

    /// Apply continuous zoom from held keys.
    pub fn apply_zoom_keys(&mut self, zoom_in: bool, zoom_out: bool) {
        if zoom_in {
            self.zoom = (self.zoom * 1.02).min(50.0);
        }
        if zoom_out {
            self.zoom = (self.zoom * 0.98).max(0.1);
        }
    }

    /// Apply scroll-wheel zoom.
    pub fn apply_scroll(&mut self, scroll_y: f32) {
        self.zoom *= 1.0 + scroll_y * 0.1;
        self.zoom = self.zoom.clamp(0.1, 50.0);
    }

    /// Build the GPU uniform from current state.
    pub fn uniforms(&self) -> CameraUniforms {
        CameraUniforms {
            offset: self.offset,
            zoom: self.zoom,
            _pad: 0.0,
        }
    }
}
