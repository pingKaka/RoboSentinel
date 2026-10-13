# Data layout

```text
data/
├── raw/<task>/<episode>/{external.mp4,wrist.mp4}
├── references/<task>/<reference>/{external.mp4,wrist.mp4}
├── annotations/episodes.jsonl
├── processed/<episode>/{reference_*.jpg,target_*.jpg,metadata.json}
└── sft/{train.jsonl,val.jsonl,test.jsonl,manifest.jsonl}
```

One annotation line represents one independent robot execution. `reason` must
describe visible evidence rather than inferred controller state.

Allowed labels and their required relationship:

| outcome | failure_type |
|---|---|
| `success` | `none` |
| `failure` | `grasp_failure`, `wrong_object`, `task_stalled`, or `target_offset` |
| `unknown` | `unknown` |

Video paths are relative to the repository root. Reference and target external
and wrist videos should start at the same event and have matching durations.
