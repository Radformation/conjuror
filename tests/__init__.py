import os
import os.path as osp

import matplotlib

TEST_FILES_DIR = osp.abspath(osp.join(osp.dirname(__file__), "test_files"))

HIDE_PLOTS = True

if os.environ.get("CI") or HIDE_PLOTS:
    matplotlib.use("Agg")

DELETE_FILES = os.environ.get("DELETE_FILES", default="False").lower() in [
    "true",
    "1",
    "yes",
]
