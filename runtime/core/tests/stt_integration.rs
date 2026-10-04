use std::path::PathBuf;

#[test]
fn transcribes_known_jfk_sample() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jfk.wav");
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/ggml-base.en-q5_1.bin");

    if !model.exists() {
        eprintln!("skipping: model not found at {model:?}, place a ggml model file there first");
        return;
    }

    let mut reader = hound::WavReader::open(&fixture).expect("jfk.wav should be readable");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "jfk.wav is expected to already be 16kHz mono");
    assert_eq!(spec.channels, 1);

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let engine = auralis_runtime::stt::SttEngine::load(&model).expect("model loads");
    let text = engine.transcribe(&samples).expect("transcription succeeds");

    let lower = text.to_lowercase();
    assert!(
        lower.contains("country"),
        "expected the well-known JFK line to mention 'country', got: {text:?}"
    );
}
