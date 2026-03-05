from enum import Enum


class DependencySliceUnparseEnum(str, Enum):
    FULL = "full"
    MINIMAL = "minimal"


class DependencySlicer:
    """Stub for dependency slicer."""

    def __init__(self, *args, **kwargs):
        pass

    def slice(self, *args, **kwargs):
        return []
