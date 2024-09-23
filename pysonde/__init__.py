from pdm.backend.hooks.version import get_version_from_scm

__version__ = str(get_version_from_scm(".").version)
