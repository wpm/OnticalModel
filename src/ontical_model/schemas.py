"""Common Pydantic schemas for structured output."""

from typing import Annotated

from pydantic import BaseModel


class Colors(BaseModel):
    """Schema for extracting colors from text."""

    colors: Annotated[set[str], "All the colors mentioned in the reply"]
