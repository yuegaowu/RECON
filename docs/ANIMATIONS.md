# RealAccident demonstrations

Nine selected collision cases from the full Waymo-trained RECON model. The
display numbers follow the order in the README. These are qualitative examples.

| Case | Clip duration | Animation |
|---|---:|---|
| 0 | 7.4 s | [Play GIF](../assets/animations/collision_03.gif) |
| 1 | 9.2 s | [Play GIF](../assets/animations/collision_06.gif) |
| 2 | 2.6 s | [Play GIF](../assets/animations/collision_09.gif) |
| 3 | 9.2 s | [Play GIF](../assets/animations/collision_12.gif) |
| 4 | 9.2 s | [Play GIF](../assets/animations/collision_14.gif) |
| 5 | 9.2 s | [Play GIF](../assets/animations/collision_18.gif) |
| 6 | 3.5 s | [Play GIF](../assets/animations/collision_25.gif) |
| 7 | 9.2 s | [Play GIF](../assets/animations/collision_27.gif) |
| 8 | 5.6 s | [Play GIF](../assets/animations/collision_31.gif) |

Orange marks the target vehicle and its highest confidence prediction. Blue
marks the collision partner, gray the surrounding traffic, and the dashed dark
curve the target’s labeled future. Only the target’s prediction is shown.

- Playback is **10 Hz**, with saved forecasts updated every **1 second**.
- Vehicles appear only when valid observations are available.
- Each forecast originally contains six 8-second candidates. The displayed
  candidate is selected by confidence, not by its error against the future.
- Elapsed points disappear every 0.1 seconds. Predictions stop at the available
  future labels; shorter clips do not display an unobserved 8-second future.
- Ground truth is shown for comparison and is not a future input to the model.

The animations reuse saved predictions from the paper’s full Waymo-trained
RECON checkpoint at epoch 95. Scene trajectories retain the dataset’s existing
interpolation and annotation corrections. The model and source trajectories
were not changed to produce these examples.

This repository includes the rendered GIFs and PNG previews. Visualization
generation code and the full review collection are maintained separately.
