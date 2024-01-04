"""Make inout into a package."""
from .xyz import configuration_from_xyz_file, read_xyz_file

__all__ = ["configuration_from_xyz_file", "read_xyz_file"]
