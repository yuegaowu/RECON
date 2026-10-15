# Source acknowledgments

This code is based on UniTraj by Lan Feng and collaborators. The upstream `LICENSE` notice and full AGPL text are retained; source-level copyright headers remain in the model implementations. The package metadata uses the license stated by that notice, rather than the inconsistent Apache label in the inherited setup script.

The baseline implementations retain their original names and research attribution: AutoBots, MTR, Wayformer, Forecast-MAE, GameFormer, SeNeVA, EMP, BeTopNet, SMART and TrajFlow. They are comparison methods, not newly proposed RECON modules. RECON's implementation reuses the Wayformer encoder/decoder and UniTraj data and training infrastructure.

ScenarioNet, MetaDrive, PyTorch, Lightning and other dependencies are installed separately and retain their own licenses. Dataset access is governed by each dataset provider. No upstream binary extensions, pretrained weights, datasets or learned dictionaries are bundled in this source release.
