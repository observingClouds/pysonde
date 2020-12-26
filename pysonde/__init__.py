# Create __version__ attribute from setup.py information

from ._version import get_versions

__import__("pkg_resources").declare_namespace(__name__)

try:
    __version__ = get_versions()["version"]
except (ValueError, ImportError):
    __version__ = "--"
