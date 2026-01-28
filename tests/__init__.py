import os
import os.path as osp

import plotly.io as pio

TEST_FILES_DIR = osp.abspath(osp.join(osp.dirname(__file__), "test_files"))

HIDE_PLOTS = os.environ.get("HIDE_PLOTS", default="True").lower() in [
    "true",
    "1",
    "yes",
]

if os.environ.get("CI") or HIDE_PLOTS:
    pio.renderers.default = None

DELETE_FILES = os.environ.get("DELETE_FILES", default="False").lower() in [
    "true",
    "1",
    "yes",
]
