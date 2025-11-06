from pydantic import BaseModel
from typing import Literal


class DataSchema(BaseModel):
    year: float
    mileage_km: float
    engine_capacity: float
    engine_power: float
    mixed_drive_fuel_consumption: float
    options: float

    brand: str
    engine_type: Literal["бензин", "дизель", "электро", "другой"]
    transmission_type: str
    interior_material: str
    body_type: Literal[
        "внедорожник", "седан", "универсал", "хэтчбек", "минивэн", "лифтбек", "купе", "другой"
    ]
    drive_type: str

    # price_usd_bin: int
