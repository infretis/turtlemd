"""Make inout into a package."""
from .xyz import particles_from_xyz_file, read_xyz_file

__all__ = ["particles_from_xyz_file", "read_xyz_file"]
