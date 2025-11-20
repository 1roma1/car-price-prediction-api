from fastapi import FastAPI
from dotenv import load_dotenv

from schema import DataSchema
from utils import load_configuration
from model import Model

load_dotenv()

app = FastAPI()
config = load_configuration("config.yaml")
model = Model(config["model_name"], config["log_transform"])
model.load(
    run_id=config["run_id"],
    estimator_name=config["estimator_name"],
    estimator_model_name=config["estimator_model_name"],
    transformer_name=config["transformer_name"],
    transformer_model_name=config["transformer_model_name"],
)


@app.post("/predict")
def predict(car_ad: DataSchema):
    price = model.predict(car_ad)
    return {"predict": int(price)}
