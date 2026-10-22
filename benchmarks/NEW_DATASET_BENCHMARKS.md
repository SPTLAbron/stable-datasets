# PlantVillage and Intel benchmark integration

Both datasets are registered for the existing RGB ViT benchmark. All seven model
YAML files include dataset entries, initially following the Imagenette budgets.
Those defaults are integration settings, not claims of completed experiments:

| Model | Default epochs | Batch per step | Accumulation | LR |
| --- | ---: | ---: | ---: | ---: |
| SimCLR / NNCLR | 150 | 256 | 1 | 0.0003 |
| Barlow Twins | 150 | 256 | 1 | 0.00015 |
| DINO / LeJEPA | 188 | 32 | 8 | 0.0005 |
| MAE | 300 | 256 | 1 | 0.00015 |
| Supervised | 150 | 128 | 1 | 0.0005 |

Completed validation used **SimCLR**, batch32: **Intel50 epochs/seed0** and
**PlantVillage10 epochs/seeds0,1**. See the
[results, curves and reproduction commands](../docs/source/datasets/experiments/plantvillage_intel/REPORT.md).

`training.scheduler_epochs` optionally preserves a longer learning-rate horizon
when `training.max_epochs` sets a shorter training budget. It must be at least the
training budget. The runner writes `final.ckpt` after normal completion.
For uninterrupted per-epoch saves with the installed Lightning, use
`checkpoint.monitor=null checkpoint.every_n_epochs=1`: its monitored top-k path
otherwise saves `last.ckpt` only when the monitored score qualifies.
