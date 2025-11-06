import tempfile
import requests
import onnxruntime
import numpy as np

from pathlib import Path
from schema import DataSchema


class Model:
    URL_TEMPLATE = "http://{host}:{port}/api/2.0/mlflow-artifacts/artifacts/{experiment_id}/models/{model_id}/artifacts/{model}"

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

    def _download_model(self, url, path):
        resp = requests.get(url)
        if resp.ok:
            with open(path, "wb") as f:
                f.write(resp.content)

    def load(
        self,
        host,
        port,
        experiment_id,
        estimator_id,
        estimator_name,
        transformer_id=None,
        transformer_name=None,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            if transformer_id:
                transformer_url = Model.URL_TEMPLATE.format(
                    host=host,
                    port=port,
                    experiment_id=experiment_id,
                    model_id=transformer_id,
                    model=transformer_name,
                )
                self._download_model(transformer_url, Path(tmp_dir, "transformer.onnx"))
                self.transformer = onnxruntime.InferenceSession(
                    Path(tmp_dir, "transformer.onnx"), providers=["CPUExecutionProvider"]
                )

            estimator_url = Model.URL_TEMPLATE.format(
                host=host,
                port=port,
                experiment_id=experiment_id,
                model_id=estimator_id,
                model=estimator_name,
            )
            self._download_model(estimator_url, Path(tmp_dir, "estimator.onnx"))

            self.estimator = onnxruntime.InferenceSession(
                Path(tmp_dir, "estimator.onnx"), providers=["CPUExecutionProvider"]
            )
