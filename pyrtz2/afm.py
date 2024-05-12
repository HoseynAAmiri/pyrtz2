import copy
import numpy as np
import os
from . import asylum


class AFM():
    def __init__(
        self,
            path: str,
            exp_name: str,
            labels: list[str],
            probe_diameter: float,
    ) -> None:

        self.path = path
        self.exp_name = exp_name
        self.labels = labels
        self.probe_diameter = probe_diameter

        self.exp_path = os.path.join(self.path, self.exp_name)
        self.load_experiment()
        self.backup_experiment()

    def load_experiment(self) -> None:
        self.experiment = asylum.load_curveset_ibw(
            self.exp_path, self.labels)
        self.curve_keys = self.experiment.keys()

    def backup_experiment(self) -> None:
        self._experiment = copy.deepcopy(self.experiment)
        self._curve_keys = copy.deepcopy(self._experiment.keys())

    def restore_experiment(self) -> None:
        self.experiment = copy.deepcopy(self._experiment)
        self.curve_keys = copy.deepcopy(self._curve_keys)

    def get_key_by_num(self, num: int, label: int = 1, fill: int = 2) -> list[tuple]:
        num_str = str(num).zfill(fill)

        keys = []
        for key in self.curve_keys:
            key_index = key[label]
            if num_str in key_index:
                keys.append(key)
        return keys

    def get_key_by_name(self, name: str, label: int = 1) -> list[tuple]:
        keys = []
        for key in self.curve_keys:
            key_index = key[label]
            if name in key_index:
                keys.append(key)
        return keys

    def drop_curves_by_num(self, drop_idx: list[int], label: int = 1, fill: int = 2) -> None:
        for idx in drop_idx:
            keys = self.get_key_by_num(idx, label=label, fill=fill)
            for key in keys:
                self.experiment.remove_curve(key)

    def compile(self, export_path: str) -> None:
        self.experiment.set_cp_by_annotations()
        self.experiment.set_vd_by_annotations()
        self.experiment.remove_unannotated()
        for idx, curve in enumerate(self.experiment):
            curve.adjust_to_contact()
            approach = curve.get_indent()
            ind = approach['ind'].to_numpy()
            dwell = curve.get_dwell()
            t_dwell = dwell['t'].to_numpy()
            directory = os.path.join(export_path, self.exp_name)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            file_name = 'key' + str(idx) + '.npz'
            np.savez(
                os.path.join(directory, file_name),
                approach=approach,
                I=ind[-1],
                dwell=dwell,
                L=t_dwell[-1] - t_dwell[0],
            )
