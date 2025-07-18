from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:   # editable install
    __version__ = "0.0.0+dev"

from algorithms.Pure_NOMAD import Pure_NOMAD  # re‑expo