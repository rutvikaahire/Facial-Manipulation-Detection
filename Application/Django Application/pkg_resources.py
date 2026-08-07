from importlib import resources
from pathlib import Path


def resource_filename(package_or_requirement, resource_name):
    traversable = resources.files(package_or_requirement).joinpath(resource_name)
    try:
        return str(Path(traversable))
    except TypeError:
        with resources.as_file(traversable) as path:
            return str(path)