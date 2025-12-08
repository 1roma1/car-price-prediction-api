from pydantic import BaseModel
from typing import Literal


class DataSchema(BaseModel):
    year: float
    mileage_km: float
    engine_capacity: float
    engine_power: float
    mixed_drive_fuel_consumption: float

    options: str
    brand: str
    engine_type: Literal["бензин", "дизель", "электро", "другой"]
    transmission_type: Literal["механика", "автомат", "вариатор", "робот"]
    interior_material: Literal[
        "ткань",
        "натуральная кожа",
        "комбинированные материалы",
        "искусственная кожа",
        "велюр",
        "алькантара",
    ]
    body_type: Literal[
        "внедорожник",
        "седан",
        "универсал",
        "хэтчбек",
        "минивэн",
        "лифтбек",
        "купе",
        "другой",
    ]
    drive_type: Literal[
        "передний привод",
        "подключаемый полный привод",
        "постоянный полный привод",
        "задний привод",
    ]
