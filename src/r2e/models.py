from pydantic import BaseModel
from typing import Optional


class Identifier(BaseModel):
    name: str = ""
    file_name: str = ""


class Function(Identifier):
    pass


class Class(Identifier):
    pass


class Module(Identifier):
    pass


class File(Identifier):
    pass


class Repo(BaseModel):
    name: str = ""
    path: str = ""
