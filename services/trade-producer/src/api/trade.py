from pydantic import BaseModel


class Trade(BaseModel):
    """
    A class to represent a trade event
    """

    product_id: str
    quantity: float
    price: float
    timestamp_ms: int
