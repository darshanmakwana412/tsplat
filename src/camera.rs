use glam::{Mat4, Quat, Vec3};

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

fn view_right_world(view_fwd: Vec3, yaw_fallback: f32) -> Vec3 {
    let right = view_fwd.cross(Vec3::Y);
    if right.length_squared() < 1e-12 {
        let (sy, cy) = yaw_fallback.sin_cos();
        Vec3::new(cy, 0.0, -sy)
    } else {
        right.normalize()
    }
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

    pub fn intrinsics(&self) -> (f32, f32, f32, f32) {
        let fy = 0.5 * self.height as f32 / (0.5 * self.fov_y).tan();
        let fx = fy;
        let cx = self.width as f32 * 0.5;
        let cy = self.height as f32 * 0.5;
        (fx, fy, cx, cy)
    }

    pub fn orbit(&mut self, dyaw: f32, dpitch: f32) {
        if dyaw.abs() < 1e-20 && dpitch.abs() < 1e-20 {
            return;
        }
        let lim = 89.0_f32.to_radians();
        let max_y = lim.sin();

        let eye = self.position();
        let mut dir = eye - self.target;
        let len_sq = dir.length_squared();
        if len_sq < 1e-20 {
            return;
        }
        dir /= len_sq.sqrt();

        let view_fwd = -dir;
        let up = view_right_world(view_fwd, self.yaw)
            .cross(view_fwd)
            .normalize();
        if dyaw != 0.0 {
            dir = Quat::from_axis_angle(up, dyaw).mul_vec3(dir).normalize();
        }

        let view_fwd = -dir;
        let right = view_right_world(view_fwd, self.yaw);
        if dpitch != 0.0 {
            dir = Quat::from_axis_angle(right, dpitch)
                .mul_vec3(dir)
                .normalize();
        }

        let y = dir.y.clamp(-max_y, max_y);
        let xz_sq = dir.x * dir.x + dir.z * dir.z;
        if xz_sq > 1e-12 {
            let xz_scale = ((1.0 - y * y).max(0.0) / xz_sq).sqrt();
            dir = Vec3::new(dir.x * xz_scale, y, dir.z * xz_scale);
        } else {
            let (sy, cy) = self.yaw.sin_cos();
            let xz = (1.0 - y * y).max(0.0).sqrt();
            dir = Vec3::new(cy * xz, y, sy * xz);
        }

        let pitch = dir.y.clamp(-1.0, 1.0).asin();
        self.pitch = pitch;
        self.yaw = if pitch.cos().abs() > 1e-5 {
            dir.x.atan2(dir.z)
        } else {
            self.yaw
        };
    }

    pub fn zoom(&mut self, factor: f32) {
        self.radius = (self.radius * factor).max(1e-3);
    }

    pub fn pan(&mut self, dx: f32, dz: f32) {
        let eye = self.position();
        let mut view_fwd = self.target - eye;
        let len_sq = view_fwd.length_squared();
        if len_sq < 1e-20 {
            return;
        }
        view_fwd /= len_sq.sqrt();
        let right = view_right_world(view_fwd, self.yaw);
        self.target += right * dx + view_fwd * dz;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}
