"""Check the installed artifact, not a source-only version claim."""
from importlib.metadata import distribution

import segrobust


def test_runtime_version_matches_installed_distribution():
    assert segrobust.__version__ == distribution('segrobust').version


def test_distribution_exposes_documented_console_entrypoint():
    entries = [entry for entry in distribution('segrobust').entry_points
               if entry.group == 'console_scripts' and entry.name == 'segrobust']
    assert len(entries) == 1
    assert entries[0].value == 'segrobust.cli:main'
    from segrobust.cli import main
    assert entries[0].load() is main
