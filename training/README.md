# Auralis training (M0 to M2)

Python research code. Run everything from `training/`.

## Tests

    python -m pytest -q

## Build a corpus and train

    cd ../data
    python -m auralis_data pull openslr dev-clean --resource 12 --dest ~/auralis-data/librispeech
    python -m auralis_data import librispeech ~/auralis-data/librispeech/LibriSpeech/dev-clean --id librispeech-dev-clean --language en --revision dev-clean --manifests ~/auralis-data/manifests
    cd ../training
    python -m auralis_stt.train configs/librispeech_tiny.json

The corpus config, a JSON file with `name` and `datasets`, is passed to `python -m auralis_data build`.

## Notes

- Training uses CUDA automatically when a CUDA build of PyTorch is installed.
- Validation is split by speaker so no speaker appears in both sets.
- Checkpoints hold weights, config, tokenizer, optimizer state, the corpus manifest and the git commit.
