# RealAccident demonstrations

Nine selected collision cases from the full Waymo-trained RECON model. The
display numbers follow the order in the README. These are qualitative examples.

| Case | Clip duration | Animation |
|---|---:|---|
| 0 | 7.4 s | [Play GIF](../assets/animations/case_0.gif) |
| 1 | 9.2 s | [Play GIF](../assets/animations/case_1.gif) |
| 2 | 2.6 s | [Play GIF](../assets/animations/case_2.gif) |
| 3 | 9.2 s | [Play GIF](../assets/animations/case_3.gif) |
| 4 | 9.2 s | [Play GIF](../assets/animations/case_4.gif) |
| 5 | 9.2 s | [Play GIF](../assets/animations/case_5.gif) |
| 6 | 3.5 s | [Play GIF](../assets/animations/case_6.gif) |
| 7 | 9.2 s | [Play GIF](../assets/animations/case_7.gif) |
| 8 | 5.6 s | [Play GIF](../assets/animations/case_8.gif) |

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
