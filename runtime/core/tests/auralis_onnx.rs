use auralis_runtime::auralis::AuralisEngine;
use auralis_runtime::stt::{load_engine, Speech};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/auralis_tiny").join(name)
}

fn read_f32(path: PathBuf) -> Vec<f32> {
    std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

#[test]
fn rust_inference_matches_pytorch_on_the_fixture() {
    let engine = AuralisEngine::load(&fixture("auralis.onnx")).expect("model loads");
    let wave = read_f32(fixture("sample.f32"));
    let expected = read_f32(fixture("expected_log_probs.f32"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture("expected.json")).unwrap()).unwrap();

    let (got, steps, vocab) = engine.log_probs(&wave).expect("inference runs");
    assert_eq!(steps as u64, meta["steps"].as_u64().unwrap());
    assert_eq!(vocab as u64, meta["vocab"].as_u64().unwrap());
    assert_eq!(got.len(), expected.len());

    let mut worst = 0f32;
    let mut sum = 0f32;
    for (a, b) in got.iter().zip(&expected) {
        let d = (a - b).abs();
        worst = worst.max(d);
        sum += d;
    }
    let mean = sum / got.len() as f32;
    println!("rust vs pytorch log-probs: max |err| = {worst:.3e}, mean |err| = {mean:.3e}");
    assert!(worst < 1e-4, "max abs error {worst:e}");
    assert!(mean < 1e-5, "mean abs error {mean:e}");

    let text = engine.decode_ids(&engine.greedy_ids(&got, steps, vocab));
    assert_eq!(text, meta["text"].as_str().unwrap());
}

#[test]
fn the_engine_is_reachable_through_the_speech_trait_and_the_factory() {
    let engine = load_engine(&fixture("auralis.onnx")).expect("factory picks the ONNX engine");
    assert_eq!(engine.engine_name(), "auralis-onnx");
    assert_eq!(engine.languages(), vec!["en".to_string()]);
    let wave = read_f32(fixture("sample.f32"));
    let segments = engine.segments(&wave, None).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text.trim(), "b d f a c");
}

#[test]
fn long_audio_is_split_into_chunks_with_offsets() {
    let mut engine = AuralisEngine::load(&fixture("auralis.onnx")).unwrap();
    engine.set_chunk_seconds(1.0);
    let wave = read_f32(fixture("sample.f32"));
    let segments = engine.segments(&wave, None).unwrap();
    assert!(segments.len() >= 2, "expected several chunks, got {}", segments.len());
    assert!(segments.windows(2).all(|w| w[1].start_s > w[0].start_s));
    assert!((segments[1].start_s - 1.0).abs() < 1e-9);
}

#[test]
fn very_short_audio_gives_no_segments_instead_of_an_error() {
    let engine = AuralisEngine::load(&fixture("auralis.onnx")).unwrap();
    assert!(engine.segments(&[0.0; 100], None).unwrap().is_empty());
}
