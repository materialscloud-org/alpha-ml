"""Request/response models of the API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ModelName = Literal[
    "ccsd",
    "b3lyp",
    "scan0",
    "delta-b3lyp-ccsd",
    "delta-scan0-ccsd",
]


class PredictRequest(BaseModel):
    symbols: list[str] = Field(description="Element symbols, one per atom")
    positions: list[list[float]] = Field(description="Cartesian positions in Å")
    model: ModelName

    @field_validator("positions")
    @classmethod
    def _three_coordinates(cls, value):
        for position in value:
            if len(position) != 3:
                raise ValueError("Each position must have exactly 3 coordinates")
        return value

    @model_validator(mode="after")
    def _same_length(self):
        if len(self.symbols) != len(self.positions):
            raise ValueError("symbols and positions must have the same length")
        return self


class PredictResponse(BaseModel):
    atomic_polarizabilities: list[list[float]] = Field(
        description="9 Cartesian components per atom, row-major"
    )
    molecular_polarizability: list[float] = Field(description="9 Cartesian components")
    compute_time: float
