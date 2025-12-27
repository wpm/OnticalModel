"""Common Pydantic schemas for structured output."""

from typing import Annotated, Optional

from pydantic import BaseModel


class NameAgeOccupation(BaseModel):
    """Schema for extracting name, age, and occupation from conversation."""

    name: Annotated[Optional[str], "The person's name if mentioned"]
    age: Annotated[Optional[int], "The person's age if mentioned"]
    occupation: Annotated[Optional[str], "The person's occupation if mentioned"]
