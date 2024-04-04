import ast
import json
import pandas as pd
from io import BytesIO

from ...afm import AFM


def process_experiment(experiment: AFM, cp_data: str, vd_data: str, indentation: float | list[float]) -> tuple[AFM, pd.DataFrame]:
    cp_annotations = json.loads(cp_data)
    cp_annotations = {ast.literal_eval(
        key): value for key, value in cp_annotations.items()}

    vd_annotations = json.loads(vd_data)
    vd_annotations = {ast.literal_eval(
        key): value for key, value in vd_annotations.items()}

    experiment.experiment.update_annotations(cp_annotations)
    experiment.experiment.update_annotations(vd_annotations)

    df = experiment.experiment.get_fit_all(
        experiment.probe_diameter, ind=indentation)
    return experiment, df


def process_indentation(indentation: str) -> float | list[float]:
    indentation_list = indentation.split(';')
    ind = [float(i) for i in indentation_list]
    if len(ind) == 1:
        ind = ind[0]
    return ind


def get_pdf(experiment: AFM) -> BytesIO:
    pdf_merger = experiment.experiment.export_figures()

    pdf_bytes = BytesIO()
    pdf_merger.write(pdf_bytes)
    pdf_bytes.seek(0)

    return pdf_bytes
