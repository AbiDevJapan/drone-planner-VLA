"""Shared validated scalar types. Cross-boundary structs forbid unexpected keys."""
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

Id = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")]
Probability = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveWeight = Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
Scalar = StrictBool | StrictInt | StrictFloat | StrictStr
