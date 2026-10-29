"""ITK exposes its generated C++ templates through lazy, runtime attributes."""

from typing import Any

def __getattr__(name: str) -> Any: ...
