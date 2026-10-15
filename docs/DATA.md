# Data interface

This release supplies conversion and loading code only. Obtain datasets through their providers and follow their access terms. Neither raw data nor processed scenario files are included. RealAccident is also not included.

The runtime reads ScenarioNet scenarios and its `dataset_summary.pkl` and mapping metadata. Scenarios contain tracks with observed/future states and validity masks, map geometry, timestamps and target metadata. Collision evaluation additionally requires the accident participants and collision timing metadata used by the shared reader. Renaming an arbitrary pickle file does not make it compatible.

## Expected layout

Set `RECON_DATA_ROOT` to the parent containing:

```text
waymo/waymo_scenarionet/training/
waymo/waymo_scenarionet/validation/
highd/waymo_scenarionet/training/
highd/waymo_scenarionet/validation/
deepaccident/scenarionet/normal/
deepaccident/scenarionet/collision_vehicle/
realaccident/scenarionet/collision_vehicle/
realaccident/scenarionet/single_vehicle/
```

The default local root is `data/` at the repository root; it is excluded from Git. Only create paths for data you actually have, and remove other entries from `eval.yaml`. The reader checks configured training and validation paths even during evaluation, so configure an accessible normal training split as well.

## Conversion entry points

Run converters from the repository root. These commands display their arguments:

```bash
python datasets/waymo_converter/prepare_data.py --help
python datasets/levelx_converter/convert_levelx_to_scenarionet.py --help
python datasets/levelx_converter/split_levelx_scenarionet.py --help
python -m datasets.deepaccident_converter.convert_deepaccident --help
```

- Waymo: `datasets/waymo_converter/prepare_data.py`. Raw TFRecords require the separate preprocessing environment in `requirements_preprocess.txt`. The second conversion stage also accepts previously processed MTR scenario files.
- highD: `datasets/levelx_converter/convert_levelx_to_scenarionet.py` and `split_levelx_scenarionet.py`. Split by recording before building training/evaluation inputs.
- DeepAccident: `datasets/deepaccident_converter/convert_deepaccident.py`. Conversion requires the user's source data and map resources.

These tools do not download or redistribute data. The source release does not package the RealAccident collection/annotation pipeline or proprietary radar records. Its evaluator can process locally supplied compatible accident scenarios.

## Derived model assets

Normal prototypes, clustered endpoints and SMART token dictionaries are excluded along with checkpoints. The prototype and agent-token builders and endpoint-clustering helper are included. SMART's map token dictionary must be supplied separately in the format expected by its model; this release does not provide a map-token generator. For evaluation, use the same derived assets as the checkpoint being evaluated.

The endpoint and agent-token builders can start without a fallback file when each object category has enough normal training samples. If a category lacks samples, they require a compatible dictionary that you generated from other normal data, and report the missing category explicitly. They do not silently create random anchors or tokens. In particular, a vehicle-only highD split cannot supply pedestrian and cyclist dictionaries.
