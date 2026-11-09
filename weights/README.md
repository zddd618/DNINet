# Released model states

These two files contain only DNINet model_state_dict tensors. Optimizer and mixed-precision scaler states were removed from the original training checkpoints. The evaluation script accepts these files directly.

| File | Source checkpoint | Source epoch | SHA-256 of released file |
| --- | --- | ---: | --- |
| kitti_ema11.pt | checkpoint/aaa-flip-spn/checkpoint_ema11.pt | 10 | 7aca6ccf51b90e17c979367da93874eefb2d88858b13b2683a54709a5fe7074e |
| nyu_ema.pt | code/aaa-nyu-loss/checkpoint/checkpoint_ema.pt | 19 | 328bd3314abd885d68902b49c79bc3a3c61e1076949300b5a1f58f074053f458 |

The KITTI submission script loaded the source checkpoint_ema11.pt and saved depth PNGs in saved/submit. Those PNGs are predictions, not weights. A local three-image replay showed small pixel-level differences from the archived PNGs, so the checkpoint provenance is established by the script but byte-for-byte reproduction of the archived submission is not claimed.

The NYUv2 model-only state was evaluated locally on 654 validation images with a single forward pass: RMSE 89.74 mm and MAE 33.18 mm under the released evaluator's protocol. These values are not presented as a reproduction of every number in the manuscript, which may use a different evaluation setting.
