from pydantic import BaseModel
from typing import Optional

class ApplicantData(BaseModel):
    class Config:
        extra = "allow"