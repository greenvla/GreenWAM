# SPDX-License-Identifier: OpenMDW-1.1
"""Read native Edge artifact identity without importing the model runtime."""
import json
import os


def is_cosmos3_edge_native_snapshot(model_dir: str) -> bool:
    """Recognize renewed Edge metadata; old remote-code snapshots remain false."""
    if not os.path.isdir(model_dir):
        return False
    for filename, key, expected in (
        ("config.json", "model_type", "cosmos3_edge"),
        ("processor_config.json", "processor_class", "Cosmos3EdgeProcessor"),
    ):
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            try:
                with open(path) as handle:
                    if json.load(handle).get(key) == expected:
                        return True
            except (OSError, ValueError):
                pass
    return False
