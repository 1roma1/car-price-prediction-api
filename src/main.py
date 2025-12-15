from fastapi import FastAPI
from dotenv import load_dotenv

from src.schema import DataSchema
from src.utils import load_configuration
from src.model import Model

load_dotenv()

# comment for ci/cd testing
app = FastAPI(root_path="/car-price-api")
config = load_configuration("config.yaml")
model = Model(config["model_name"], config["log_transform"])
model.load(
    run_id=config["run_id"],
    estimator_name=config["estimator_name"],
    estimator_model_name=config["estimator_model_name"],
    transformer_name=config["transformer_name"],
    transformer_model_name=config["transformer_model_name"],
)


@app.get("/health")
def check_health():
    return {"status": "ok"}


@app.post("/predict")
def predict(car_ad: DataSchema):
    price = model.predict(car_ad)
    return {"predicted_price": int(price)}
