import aceai
from aceai import config


def test_package_imports():
    assert aceai.__version__


def test_paths_point_into_repo():
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()
    assert config.DATA_RAW == config.PROJECT_ROOT / "data" / "raw"
