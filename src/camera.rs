use glam::{Mat4, Vec3};

/// Orbit camera that circles a target point on a sphere.
///
/// `width` / `height` are pixel dimensions of the framebuffer (i.e. terminal
/// columns for width, 2 * terminal rows for height with half-block rendering).
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub fov_y: f32,
    pub width: u32,
    pub height: u32,
    pub znear: f32,
    pub zfar: f32,
}

impl OrbitCamera {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            radius: 5.0,
            fov_y: 60.0_f32.to_radians(),
            width,
            height,
            znear: 0.05,
            zfar: 1000.0,
        }
    }

    pub fn position(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let dir = Vec3::new(cp * sy, sp, cp * cy);
        self.target + dir * self.radius
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.target, Vec3::Y)
    }

    /// Returns `(fx, fy, cx, cy)` intrinsics. Half-block pixels are roughly
    /// square, so `fx == fy`.
    pub fn intrinsics(&self) -> (f32, f32, f32, f32) {
        let fy = 0.5 * self.height as f32 / (0.5 * self.fov_y).tan();
        let fx = fy;
        let cx = self.width as f32 * 0.5;
        let cy = self.height as f32 * 0.5;
        (fx, fy, cx, cy)
    }

    pub fn orbit(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        let lim = 89.0_f32.to_radians();
        self.pitch = (self.pitch + dpitch).clamp(-lim, lim);
    }

    pub fn zoom(&mut self, factor: f32) {
        self.radius = (self.radius * factor).max(1e-3);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}
