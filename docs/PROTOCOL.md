# Shared training and evaluation conventions

The release keeps the research `train.py`, `benchmark_eval.py`, summary generator and shared data processing. Each model retains its input adapter, losses, optimizer and method YAML. A shared framework does not mean that every architecture uses identical optimization settings.

## Inputs and training

- Normal driving data is used for training and validation. Accident subsets are for evaluation.
- Common temporal input: 11 historical samples and 80 future samples at 10 Hz.
- RECON and the multimodal baselines predict candidate trajectories and scores. SMART provides a single trajectory reference; its metrics are treated accordingly by the evaluator.
- `train_batch_size` and `eval_batch_size` count scenes per GPU. After expanding scenes into prediction targets, `max_agents_per_batch` caps the number of targets.
- `prediction_targets=official` uses the dataset target metadata and shared filtering. Accident entries in `eval.yaml` select `crash` targets. The default does not add the SDC as another target.
- `max_data_num` limits the initial scene pool. `limit_train_batches` consumes a fraction of shuffled training batches each epoch. `limit_val_batches` controls validation batches. They are different settings.
- Checkpoint selection uses normal validation metrics. The selected checkpoint is then reused across target datasets.

## Evaluation

`eval.yaml` inherits `config.yaml` and selects one method YAML. It keeps the original unified evaluation route (`strict_eval_support: false`) and reads ScenarioNet on the fly. Its normal subsets are capped at 5,000 Waymo and 3,000 highD scenes, while accident entries use all available scenarios. SMART has its separately configured normal-scene window cap. Target caps and model-specific validity checks can affect retained records; the evaluator reports sample counts.

| Output field | Meaning |
|---|---|
| `minADE6` | Lowest average displacement error among available candidates, over valid future samples |
| `minFDE6` | Lowest endpoint error at the last valid future sample |
| `MissRate` | Fraction whose best candidate endpoint error exceeds 2 m |
| `ADE1@3s`, `ADE1@5s`, `ADE1@8s` | Average displacement error of the highest confidence candidate through the indicated horizon, subject to valid labels |
| `FDE1@3s`, `FDE1@5s`, `FDE1@8s` | Highest confidence candidate's error at that horizon |
| `top1_is_best` | Fraction whose highest confidence candidate also has the smallest endpoint error |
| `brier_minFDE6` | Best endpoint error plus the confidence penalty used by the shared evaluator |

Distances are in meters. A high `top1_is_best` does not itself mean the absolute trajectory error is small. For a short accident segment, the last valid endpoint need not be 8 seconds. Separate TTC outputs move the prediction origin to different times before impact and summarize the valid segment through impact; they are not the ordinary fixed-origin 8-second metric.

`regular_metrics.csv`, `accident_collider_metrics.csv` and `accident_ttc.csv` summarize different subsets or prediction origins. Optional saved predictions support offline analysis. No historical CSVs or prediction records are included in this release, and no metric definitions are replaced for the public package.
