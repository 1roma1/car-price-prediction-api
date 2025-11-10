import os
import base64
import tempfile
import requests
import onnxruntime
import numpy as np

from pathlib import Path
from schema import DataSchema


class Model:
    def __init__(self, model_name, log_transform=False):
        self.model_name = model_name
        self.log_transform = log_transform

    def _process(self, input: DataSchema) -> dict:
        schema = input.model_json_schema()["properties"]
        input_dict = input.model_dump()

        for key in input_dict.keys():
            if schema[key]["type"] == "number":
                input_dict[key] = np.array(input_dict[key], dtype=np.float32).reshape(1, -1)
            else:
                input_dict[key] = np.array(input_dict[key]).reshape(1, -1)
        return input_dict

    def predict(self, input: DataSchema):
        input = self._process(input)
        if self.transformer:
            transformer_label_name = self.transformer.get_outputs()[0].name
            transformer_output = self.transformer.run([transformer_label_name], input)[0]

        estimator_input_name = self.estimator.get_inputs()[0].name
        estimator_label_name = self.estimator.get_outputs()[0].name
        prediction = self.estimator.run(
            [estimator_label_name], {estimator_input_name: transformer_output.astype(np.float64)}
        )[0][0][0]
        return np.expm1(prediction) if self.log_transform else prediction

    def _download_model(self, url, path, params):
        basic_auth_str = f"{os.getenv('MLFLOW_TRACKING_USERNAME')}:{os.getenv('MLFLOW_TRACKING_PASSWORD')}".encode()
        auth_str = "Basic " + base64.standard_b64encode(basic_auth_str).decode("utf-8")
        headers = {"Authorization": auth_str, "User-Agent": "mlflow-python-client/2.22.2"}
        resp = requests.get(url, headers=headers, params=params)
        if resp.ok:
            with open(path, "wb") as f:
                f.write(resp.content)

    def load(self, run_id, estimator_name, transformer_name=None):
        url = os.getenv("MLFLOW_URL")
        with tempfile.TemporaryDirectory() as tmp_dir:
            if transformer_name:
                params = {
                    "path": f"{transformer_name}/model.onnx",
                    "run_uuid": f"{run_id}",
                }
                self._download_model(url, Path(tmp_dir, "transformer.onnx"), params)
                self.transformer = onnxruntime.InferenceSession(
                    Path(tmp_dir, "transformer.onnx"), providers=["CPUExecutionProvider"]
                )

            params = {
                "path": f"{estimator_name}/model.onnx",
                "run_uuid": f"{run_id}",
            }
            self._download_model(url, Path(tmp_dir, "estimator.onnx"), params)

            self.estimator = onnxruntime.InferenceSession(
                Path(tmp_dir, "estimator.onnx"), providers=["CPUExecutionProvider"]
            )
