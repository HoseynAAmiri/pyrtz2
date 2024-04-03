import ast
import json
import pandas as pd

from pyrtz2.afm import AFM


def process_experiment(experiment: AFM, cp_data: str, vd_data: str) -> AFM:
    cp_annotations = json.loads(cp_data)
    cp_annotations = {ast.literal_eval(
        key): value for key, value in cp_annotations.items()}

    vd_annotations = json.loads(vd_data)
    vd_annotations = {ast.literal_eval(
        key): value for key, value in vd_annotations.items()}

    # THIS CAN BE REPLACED BY UPDATE ANNOTATIONS FROM FILE
    for key, cp in cp_annotations.items():
        curve = experiment.experiment[key]
        curve.set_contact_index(cp)

    experiment.remove_unannotated()
    for key in experiment.curve_keys:
        curve = experiment.experiment[key]
        if curve.contact_index > 2 and vd_annotations[key]:
            curve.correct_virt_defl()
        experiment.adjust_to_contact(key)

    return experiment


def get_all_fit(experiment_processed: AFM, indentation: float = 1.0e-6) -> pd.DataFrame | pd.Series:
    all_fit = []
    for key in experiment_processed.curve_keys:
        fit_results = experiment_processed.get_fit(key, indentation)
        all_fit.append(fit_results)

    combined_results = pd.concat(all_fit, ignore_index=True)
    return combined_results
