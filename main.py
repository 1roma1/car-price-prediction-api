from fastapi import FastAPI

from schema import DataSchema
from utils import load_configuration
from model import Model


app = FastAPI()
config = load_configuration("config.yaml")
model = Model(config["model_name"], config["log_transform"])
model.load(
    host=config["host"],
    port=config["port"],
    experiment_id=config["experiment_id"],
    estimator_id=config["estimator_id"],
    estimator_name=config["estimator_model"],
    transformer_id=config["transformer_id"],
    transformer_name=config["transformer_model"],
)


@app.post("/predict")
def predict(car_ad: DataSchema):
    price = model.predict(car_ad)
    return {"predict": int(price)}
