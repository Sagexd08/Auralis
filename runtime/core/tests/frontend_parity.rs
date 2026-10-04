
use auralis_runtime::frontend::Frontend;

const MAX_ABS_TOL: f64 = 1e-9;
const MEAN_ABS_TOL: f64 = 1e-11;

#[test]
fn rust_log_mel_matches_python_reference() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/logmel_parity.f64");
    let bytes = std::fs::read(&path).expect("fixture missing; run training/scripts/gen_parity_fixture.py");
    let vals: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    let (signal, expected) = vals.split_at(16_000);
    assert_eq!(expected.len(), 80 * 101);

    let got = Frontend::default().log_mel(signal);
    assert_eq!((got.len(), got[0].len()), (80, 101));

    let (mut max, mut sum) = (0f64, 0f64);
    for (m, row) in got.iter().enumerate() {
        for (t, v) in row.iter().enumerate() {
            let e = (v - expected[m * 101 + t]).abs();
            max = max.max(e);
            sum += e;
        }
    }
    let mean = sum / expected.len() as f64;
    println!("rust vs python log-mel: max |err| = {max:.3e}, mean |err| = {mean:.3e}");
    assert!(max < MAX_ABS_TOL, "max abs error {max:e} exceeds {MAX_ABS_TOL:e}");
    assert!(mean < MEAN_ABS_TOL, "mean abs error {mean:e} exceeds {MEAN_ABS_TOL:e}");
}
