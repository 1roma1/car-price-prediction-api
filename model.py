import os
import base64
import tempfile
import pickle
import requests
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from pathlib import Path
from schema import DataSchema


class Model:
    def __init__(self, model_name, log_transform=False):
        self.model_name = model_name
        self.log_transform = log_transform

    def _process(self, input: DataSchema) -> dict:
        input_dict = input.model_dump()

        for key in input_dict.keys():
            input_dict[key] = [input_dict[key]]
        return pd.DataFrame(input_dict)

    def predict(self, input: DataSchema):
        input = self._process(input)
        if self.transformer:
            input = self.transformer.transform(input)
        prediction = self.estimator.predict(input)
        return np.expm1(prediction) if self.log_transform else prediction

    def _download_model(self, url, path, params):
        basic_auth_str = f"{os.getenv('MLFLOW_TRACKING_USERNAME')}:{os.getenv('MLFLOW_TRACKING_PASSWORD')}".encode()
        auth_str = "Basic " + base64.standard_b64encode(basic_auth_str).decode("utf-8")
        headers = {"Authorization": auth_str, "User-Agent": "mlflow-python-client/2.22.2"}
        resp = requests.get(url, headers=headers, params=params)
        if resp.ok:
            with open(path, "wb") as f:
                f.write(resp.content)

    def load(
        self,
        run_id: str,
        estimator_name: str,
        estimator_model_name: str,
        transformer_name: str = None,
        transformer_model_name: str = None,
    ):
        url = os.getenv("MLFLOW_URL")
        with tempfile.TemporaryDirectory() as tmp_dir:
            if transformer_name:
                params = {
                    "path": f"{transformer_name}/{transformer_model_name}",
                    "run_uuid": f"{run_id}",
                }
                self._download_model(url, Path(tmp_dir, transformer_model_name), params)
                with open(Path(tmp_dir) / transformer_model_name, "rb") as f:
                    self.transformer = pickle.load(f)

            params = {
                "path": f"{estimator_name}/{estimator_model_name}",
                "run_uuid": f"{run_id}",
            }
            self._download_model(url, Path(tmp_dir, estimator_model_name), params)

            if self.model_name == "cb":
                self.estimator = CatBoostRegressor().load_model(Path(tmp_dir, estimator_model_name))
