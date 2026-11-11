# Qwen3-VL robot failure fine-tuning

This directory separates data, parameters, and execution code:

```text
training/
├── data/       # Raw videos, human annotations, processed frames, SFT JSONL
├── config/     # Dataset, LoRA, and remote-server parameters
└── run/        # Dataset preparation, validation, training, and SSH orchestration
```

## 1. Prepare data

Copy the annotation template and fill one JSON object per robot execution:

```bash
cp training/data/annotations/episodes.example.jsonl \
  training/data/annotations/episodes.jsonl
```

Place execution videos under `training/data/raw/` and successful reference
videos under `training/data/references/`. Paths in the annotation file are
relative to the repository root.

Each episode must have an explicit `train`, `val`, or `test` split. Split by
recording session before generating frames; clips or augmented variants from
the same execution must never cross splits.

Build fixed-time-interval dual-camera image sequences and SFT files:

```bash
conda run -n robosentinel \
  python training/run/prepare_dataset.py --config training/config/data.yaml

python training/run/validate_dataset.py --config training/config/data.yaml
```

The generated model input for every episode is:

```text
label(reference 01/NN, timestamp) -> dual-camera image
...
label(reference NN/NN, timestamp) -> dual-camera image
label(target 01/MM, timestamp)    -> dual-camera image
...
label(target MM/MM, timestamp)    -> dual-camera image
prompt -> assistant JSON
```

The external camera is the upper half of each image and the wrist camera is
the lower half. `frame_sample_interval_seconds` in `training/config/data.yaml`
sets one shared temporal scale for reference and target videos; sequence lengths
depend on their durations.

## 2. Configure training

Edit `training/config/train.yaml`. At minimum, set `model` to the Qwen3-VL-8B
path that exists on the remote server. The default configuration performs LoRA
SFT, freezes the vision encoder and aligner, and uses batch size 1 with gradient
accumulation.

Preview the exact command without starting training:

```bash
python training/run/run_training.py \
  --config training/config/train.yaml --dry-run
```

## 3. Run on a remote server

Create the untracked connection file:

```bash
cp training/config/remote.env.example training/config/remote.env
```

Edit the host, user, remote project directory, and Conda environment. Then run:

```bash
# Check SSH and remote dependencies.
python training/run/remote.py check

# Upload training code and data. Checkpoints are excluded by default.
python training/run/remote.py sync

# Build/validate the dataset remotely and launch LoRA in the background.
python training/run/remote.py prepare
python training/run/remote.py train

# Inspect progress and retrieve adapters/results.
python training/run/remote.py status
python training/run/remote.py download
```

`remote.py train` writes the remote PID to `training/output/train.pid` and the
combined stdout/stderr log to `training/output/train.log`. It does not store a
password; use an SSH key or an existing SSH agent.

Before trusting a checkpoint, compare it with the unfine-tuned model on the
unchanged `test.jsonl`. Do not select checkpoints using test-set results.

The SFT target schema uses `outcome`, `failure_type`, `evidence_frames`, and
`reason`. Adapter inference must use the same `prompt/training_sft_zh.txt` protocol and
the same fixed-interval image ordering. Keep inference result handling in passthrough mode,
or update downstream consumers to understand this schema; switching back to a
different prompt or output schema discards part of the fine-tuning benefit.
