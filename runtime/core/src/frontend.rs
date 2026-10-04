
use std::f64::consts::PI;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 80;
pub const LOG_EPS: f64 = 1e-10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    Rect,
    Hann,
    Hamming,
    Blackman,
}

pub fn window(kind: Window, n: usize, periodic: bool) -> Vec<f64> {
    let d = if periodic { n as f64 } else { (n - 1) as f64 };
    (0..n)
        .map(|k| {
            let k = k as f64;
            match kind {
                Window::Rect => 1.0,
                Window::Hann => 0.5 - 0.5 * (2.0 * PI * k / d).cos(),
                Window::Hamming => 0.54 - 0.46 * (2.0 * PI * k / d).cos(),
                Window::Blackman => 0.42 - 0.5 * (2.0 * PI * k / d).cos() + 0.08 * (4.0 * PI * k / d).cos(),
            }
        })
        .collect()
}

pub fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    assert!(n.is_power_of_two() && im.len() == n, "fft length must be a power of two");
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

pub fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

pub fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

pub fn mel_filterbank(sr: usize, n_fft: usize, n_mels: usize) -> Vec<Vec<f64>> {
    let (fmin, fmax) = (0.0, sr as f64 / 2.0);
    let (m0, m1) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let edges: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let bins = n_fft / 2 + 1;
    let freqs: Vec<f64> = (0..bins).map(|k| fmax * k as f64 / (bins - 1) as f64).collect();
    (0..n_mels)
        .map(|i| {
            let (lo, mid, hi) = (edges[i], edges[i + 1], edges[i + 2]);
            freqs
                .iter()
                .map(|&f| ((f - lo) / (mid - lo)).min((hi - f) / (hi - mid)).max(0.0))
                .collect()
        })
        .collect()
}

fn reflect_pad(x: &[f64], pad: usize) -> Vec<f64> {
    let n = x.len() as isize;
    assert!(n > 1, "reflect padding needs at least two samples");
    let period = 2 * (n - 1);
    let at = |i: isize| -> f64 {
        let m = i.rem_euclid(period);
        x[(if m < n { m } else { period - m }) as usize]
    };
    (-(pad as isize)..n + pad as isize).map(at).collect()
}

pub struct Frontend {
    n_fft: usize,
    hop: usize,
    win: Vec<f64>,
    fb: Vec<Vec<f64>>,
}

impl Default for Frontend {
    fn default() -> Self {
        Self::new(SAMPLE_RATE, N_FFT, WIN_LENGTH, HOP, N_MELS)
    }
}

impl Frontend {
    pub fn new(sr: usize, n_fft: usize, win_length: usize, hop: usize, n_mels: usize) -> Self {
        let mut win = vec![0.0; n_fft];
        let off = (n_fft - win_length) / 2;
        win[off..off + win_length].copy_from_slice(&window(Window::Hann, win_length, true));
        Self { n_fft, hop, win, fb: mel_filterbank(sr, n_fft, n_mels) }
    }

    pub fn n_mels(&self) -> usize {
        self.fb.len()
    }

    pub fn power_spectrogram(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let padded = reflect_pad(x, self.n_fft / 2);
        let n_frames = 1 + (padded.len() - self.n_fft) / self.hop;
        let bins = self.n_fft / 2 + 1;
        (0..n_frames)
            .map(|t| {
                let seg = &padded[t * self.hop..t * self.hop + self.n_fft];
                let mut re: Vec<f64> = seg.iter().zip(&self.win).map(|(a, w)| a * w).collect();
                let mut im = vec![0.0; self.n_fft];
                fft(&mut re, &mut im);
                (0..bins).map(|k| re[k] * re[k] + im[k] * im[k]).collect()
            })
            .collect()
    }

    pub fn log_mel(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let power = self.power_spectrogram(x);
        self.fb
            .iter()
            .map(|row| {
                power
                    .iter()
                    .map(|frame| (row.iter().zip(frame).map(|(a, b)| a * b).sum::<f64>() + LOG_EPS).ln())
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_naive_dft() {
        let x: Vec<f64> = (0..16).map(|i| ((i * 7 % 5) as f64) - 2.0).collect();
        let (mut re, mut im) = (x.clone(), vec![0.0; 16]);
        fft(&mut re, &mut im);
        for k in 0..16 {
            let (mut sr, mut si) = (0.0, 0.0);
            for (n, v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * n) as f64 / 16.0;
                sr += v * a.cos();
                si += v * a.sin();
            }
            assert!((re[k] - sr).abs() < 1e-9 && (im[k] - si).abs() < 1e-9);
        }
    }

    #[test]
    fn shapes_and_hz_mel_roundtrip() {
        let fe = Frontend::default();
        let lm = fe.log_mel(&vec![0.0; 16_000]);
        assert_eq!((lm.len(), lm[0].len()), (80, 101));
        assert!((mel_to_hz(hz_to_mel(1234.5)) - 1234.5).abs() < 1e-9);
    }

    #[test]
    fn reflect_pad_matches_numpy() {
        assert_eq!(reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2), vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }
}
