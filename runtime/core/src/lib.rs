pub mod audio;
pub mod denoise;
pub mod resample;
pub mod pipeline;
pub mod stt;
pub mod text;
pub mod vad;

pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_crate_version() {
        assert_eq!(crate_version(), "0.1.0");
    }
}
