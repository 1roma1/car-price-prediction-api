from pydantic import BaseModel
from typing import Literal


class DataSchema(BaseModel):
    year: float
    mileage_km: float
    engine_capacity: float
    engine_power: float
    mixed_drive_fuel_consumption: float
    options: float
    price_usd_bin: float

    brand: Literal["Audi", "Mazda"]
    engine_type: Literal["бензин", "дизель", "электро", "другой"]
    transmission_type: Literal["механика", "автомат"]
    interior_material: Literal["ткань", "кожа"]
    body_type: Literal[
        "внедорожник", "седан", "универсал", "хэтчбек", "минивэн", "лифтбек", "купе", "другой"
    ]
    drive_type: Literal["передний привод", "задний привод"]
