use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use crossbeam_channel::{Receiver, Sender};

/// Owns an open input stream. Dropping this stops capture.
pub struct AudioCapture {
    _stream: Stream,
    pub sample_rate: u32,
    receiver: Receiver<f32>,
}

impl AudioCapture {
    /// Opens the default input device's default mono-compatible config and starts
    /// streaming samples immediately. Samples arrive as mono f32 at `sample_rate`
    /// (the device's native rate — callers resample as needed).
    pub fn start() -> Result<Self> {
        Self::start_with_device(None)
    }

    /// Like `start`, but opens a specific input device by name instead of the
    /// system default. `None` falls back to the default device.
    pub fn start_with_device(device_name: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => host
                .input_devices()
                .context("failed to enumerate input devices")?
                .find(|d| d.name().map(|n| n == name).unwrap_or(false))
                .with_context(|| format!("input device {name:?} not found"))?,
            None => host
                .default_input_device()
                .context("no default input (microphone) device found")?,
        };
        let config = device
            .default_input_config()
            .context("failed to read default input config")?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let sample_format = config.sample_format();

        let (tx, rx): (Sender<f32>, Receiver<f32>) = crossbeam_channel::unbounded();

        let err_fn = |err| eprintln!("audio stream error: {err}");

        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| send_mono(&tx, data, channels),
                err_fn,
                None,
            )?,
            SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    let floats: Vec<f32> = data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                    send_mono(&tx, &floats, channels)
                },
                err_fn,
                None,
            )?,
            other => anyhow::bail!("unsupported input sample format: {other:?}"),
        };

        stream.play().context("failed to start input stream")?;

        Ok(Self {
            _stream: stream,
            sample_rate,
            receiver: rx,
        })
    }

    /// Drains whatever samples have arrived so far without blocking.
    pub fn drain_available(&self) -> Vec<f32> {
        self.receiver.try_iter().collect()
    }
}

/// Lists available input device names, for a mic-selection UI. The system
/// default is whichever `cpal::default_input_device()` resolves to, which
/// may not be first in this list.
pub fn list_input_device_names() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host.input_devices().context("failed to enumerate input devices")?;
    Ok(devices.filter_map(|d| d.name().ok()).collect())
}

fn send_mono(tx: &Sender<f32>, data: &[f32], channels: usize) {
    if channels <= 1 {
        for &s in data {
            let _ = tx.send(s);
        }
        return;
    }
    for frame in data.chunks(channels) {
        let avg = frame.iter().sum::<f32>() / channels as f32;
        let _ = tx.send(avg);
    }
}
