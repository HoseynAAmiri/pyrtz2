import copy
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
