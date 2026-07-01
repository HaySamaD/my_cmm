from pydantic import BaseModel
from typing import Dict


class TypeReturn(BaseModel):
    type: str


class Parameter(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: TypeReturn
