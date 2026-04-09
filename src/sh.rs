use glam::Vec3;

/// Band-0 real spherical harmonic normalization constant (Y_0^0).
pub const SH_C0: f32 = 0.28209479177387814;

/// Convert the band-0 DC SH coefficient stored in INRIA .ply files to an RGB
/// color in [0, 1]. Higher bands are deliberately ignored for the MVP.
#[inline]
pub fn sh_band0_to_rgb(f_dc: Vec3) -> Vec3 {
    (Vec3::splat(0.5) + SH_C0 * f_dc).clamp(Vec3::ZERO, Vec3::ONE)
}
