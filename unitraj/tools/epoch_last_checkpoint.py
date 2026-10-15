"""Keep the normal top-k policy, but always checkpoint the latest completed epoch.

Lightning 2.6 can skip save_last=True when the monitored score does not improve.
This callback repairs persistence only; it does not change training or selection.
"""
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint


class EpochLastCheckpoint(ModelCheckpoint):
    def load_state_dict(self, state_dict):
        # Moving a run can leave its old directory as a compatibility symlink.
        # Lightning compares path strings and otherwise drops the saved top-k
        # scores. Normalize only proven aliases of the same existing directory.
        saved_directory = state_dict.get('dirpath')
        if saved_directory and self.dirpath and saved_directory != self.dirpath:
            old, current = Path(saved_directory), Path(self.dirpath)
            try:
                same_directory = old.is_dir() and current.is_dir() and old.samefile(current)
            except OSError:
                same_directory = False
            if same_directory:
                def relocate(path):
                    if not path:
                        return path
                    try:
                        return str(current / Path(path).relative_to(old))
                    except ValueError:
                        return path

                state_dict = dict(state_dict)  # Never rewrite the loaded checkpoint.
                state_dict['dirpath'] = self.dirpath
                for key in ('best_model_path', 'kth_best_model_path', 'last_model_path'):
                    if key in state_dict:
                        state_dict[key] = relocate(state_dict[key])
                if 'best_k_models' in state_dict:
                    state_dict['best_k_models'] = {
                        relocate(path): score for path, score in state_dict['best_k_models'].items()
                    }
        super().load_state_dict(state_dict)

    def _ensure_latest(self, trainer):
        if not self.save_last or trainer.fast_dev_run or trainer.sanity_checking:
            return
        if not self.last_model_path or self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        # All DDP ranks participate; Lightning's strategy handles the actual IO.
        self._ensure_latest(trainer)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self._ensure_latest(trainer)
