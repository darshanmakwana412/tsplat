use glam::Vec3;

pub const SH_C0: f32 = 0.28209479177387814;

#[inline]
pub fn sh_band0_to_rgb(f_dc: Vec3) -> Vec3 {
    (Vec3::splat(0.5) + SH_C0 * f_dc).clamp(Vec3::ZERO, Vec3::ONE)
}
