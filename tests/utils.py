from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import os.path as osp
import shutil
import time
from collections.abc import Sequence
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Callable, TypedDict
from urllib.request import urlopen

from google.cloud import storage
from py_linq import Enumerable
from requests import ReadTimeout

from tests import DELETE_FILES

GCP_BUCKET_NAME = "pylinac_test_files"
LOCAL_TEST_DIR = "test_files"

# make the local test dir if it doesn't exist
os.makedirs(osp.join(osp.dirname(__file__), LOCAL_TEST_DIR), exist_ok=True)


@contextlib.contextmanager
def access_gcp() -> storage.Client:
    # access GCP
    credentials_file = Path(__file__).parent.parent / "GCP_creds.json"
    # check if the credentials file is available (local dev)
    # if not, load from the env var (test pipeline)
    if not credentials_file.is_file():
        with open(credentials_file, "wb") as f:
            creds = base64.b64decode(os.environ.get("GOOGLE_CREDENTIALS", ""))
            f.write(creds)
    client = storage.Client.from_service_account_json(str(credentials_file))
    try:
        yield client
    finally:
        del client


@lru_cache
def gcp_bucket_object_list(bucket_name: str) -> list:
    with access_gcp() as storage_client:
        return list(storage_client.list_blobs(bucket_name))


def get_folder_from_cloud_repo(
    folder: list[str],
    skip_exists: bool = True,
    local_dir: str | Path = LOCAL_TEST_DIR,
    cloud_repo: str = GCP_BUCKET_NAME,
) -> str:
    """Get a folder from a GCP bucket.

    Parameters
    ----------
    folder: list[str]
        The folder to get from the GCP bucket.
    skip_exists
        If True, only checks that the destination folder exists and isn't empty.
        This is helpful for avoiding network calls since querying GCP can cost significant time.
    local_dir
        The local directory to download the files to.
    cloud_repo
        The GCP bucket to download from.
    """
    dest_folder = Path(local_dir, *folder)
    if skip_exists and dest_folder.exists() and len(list(dest_folder.iterdir())) > 0:
        return str(dest_folder)
    # get the folder data
    all_blobs = gcp_bucket_object_list(cloud_repo)
    blobs = (
        Enumerable(all_blobs)
        .where(lambda b: len(b.name.split("/")) > len(folder))
        .where(lambda b: b.name.split("/")[1] != "")
        .where(
            lambda b: all(f in b.name.split("/")[idx] for idx, f in enumerate(folder))
        )
        .to_list()
    )

    # make root folder if need be
    dest_folder = osp.join(osp.dirname(__file__), local_dir, *folder)
    if not osp.isdir(dest_folder):
        os.makedirs(dest_folder)

    # make subfolders if need be
    subdirs = [b.name.split("/")[1:-2] for b in blobs if len(b.name.split("/")) > 2]
    dest_sub_folders = [
        osp.join(osp.dirname(__file__), local_dir, *folder, *f) for f in subdirs
    ]
    for sdir in dest_sub_folders:
        if not osp.isdir(sdir):
            os.makedirs(sdir)

    for blob in blobs:
        # download file
        path = osp.join(dest_folder, blob.name.split("/")[-1])
        if not os.path.exists(path):
            blob.download_to_filename(path)

    return osp.join(osp.dirname(__file__), local_dir, *folder)


def get_file_from_cloud_test_repo(path: list[str], force: bool = False) -> str:
    """Get a single file from GCP storage. Returns the path to disk it was downloaded to"""
    local_filename = osp.join(osp.dirname(__file__), LOCAL_TEST_DIR, *path)
    if osp.isfile(local_filename) and not force:
        with open(local_filename, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            print(f"Local file found: {local_filename}@{file_hash}")
        return local_filename
    with access_gcp() as client:
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(
            str(PurePosixPath(*path))
        )  # posix because google storage is on unix and won't find path w/ windows path
        # make any necessary subdirs leading up to the file
        if len(path) > 1:
            for idx in range(1, len(path)):
                local_dir = osp.join(osp.dirname(__file__), LOCAL_TEST_DIR, *path[:idx])
                os.makedirs(local_dir, exist_ok=True)

        try:
            blob.download_to_filename(
                local_filename, timeout=120
            )  # longer timeout for large files; in the CI/CD pipeline sometimes we are bandwidth choked.
        except ReadTimeout as e:
            # GCP creates a zero-byte file to start; if we timeout or otherwise fail, the 0-size file still exists;
            # delete it so other tests don't think the file is valid
            Path(local_filename).unlink(missing_ok=True)
            raise ConnectionError(
                f"Could not download the file from GCP. Are you connected to the internet? You could also be bandwidth-limited. Try again later. {e}"
            )
        time.sleep(2)
        print(
            f"Downloaded from GCP: {local_filename}@{hashlib.md5(open(local_filename, 'rb').read()).hexdigest()}"
        )
        return local_filename


def has_www_connection():
    try:
        with urlopen("http://www.google.com") as r:
            return r.status == 200
    except Exception:
        return False


def save_file(method, *args, as_file_object=None, to_single_file=True, **kwargs):
    """Save a file using the passed method and assert it exists after saving.
    Also deletes the file after checking for existence."""
    if as_file_object is None:  # regular file
        with TemporaryDirectory() as tmpdir:
            if to_single_file:
                tmpfile = osp.join(tmpdir, "myfile")
                method(tmpfile, *args, **kwargs)
            else:
                method(tmpdir, *args, **kwargs)
    else:
        if "b" in as_file_object:
            temp = BytesIO
        elif "s" in as_file_object:
            temp = StringIO
        with temp() as t:
            method(t, *args, **kwargs)


def point_equality_validation(point1, point2):
    if point1.x != point2.x:
        raise ValueError(f"{point1.x} does not equal {point2.x}")
    if point1.y != point2.y:
        raise ValueError(f"{point1.y} does not equal {point2.y}")


class CloudFileMixin:
    """A mixin that provides attrs and a method to get the absolute location of the
    file using syntax.

    Usage
    -----
    0. Override ``dir_location`` with the directory of the destination.
    1. Override ``file_path`` with a list that contains the subfolder(s) and file name.
    """

    file_name: str | Sequence[str] | None = None
    dir_path: Sequence[str]
    delete_file = True

    @classmethod
    def get_filename(cls) -> str:
        """Return the canonical path to the file on disk. Download if it doesn't exist."""
        if isinstance(cls.file_name, (list, tuple)):
            full_path = Path(LOCAL_TEST_DIR, *cls.dir_path, *cls.file_name).absolute()
        else:
            full_path = Path(LOCAL_TEST_DIR, *cls.dir_path, cls.file_name).absolute()
        if full_path.is_file():
            return str(full_path)
        else:
            if isinstance(cls.file_name, (list, tuple)):
                return get_file_from_cloud_test_repo([*cls.dir_path, *cls.file_name])
            else:
                return get_file_from_cloud_test_repo([*cls.dir_path, cls.file_name])

    @classmethod
    def tearDownClass(cls):
        if cls.delete_file and DELETE_FILES and cls.file_name:
            file = cls.get_filename()
            if osp.isfile(file):
                os.remove(file)
            elif osp.isdir(file):
                shutil.rmtree(file)
        super().tearDownClass()


class MixinTesterBase:
    klass = Callable


class FromZipTesterMixin(MixinTesterBase):
    zip: list[str]
    zip_kwargs: dict = {}

    def test_from_zip(self):
        file = get_file_from_cloud_test_repo(self.zip)
        inst = self.klass.from_zip(file, **self.zip_kwargs)
        self.assertIsInstance(inst, self.klass)


class FromDemoImageTesterMixin(MixinTesterBase):
    demo_load_method: str = "from_demo_image"

    def test_from_demo(self):
        inst = getattr(self.klass, self.demo_load_method)()
        self.assertIsInstance(inst, self.klass)


class InitTesterMixin(MixinTesterBase):
    init_file: list[str]
    init_kwargs: dict = {}
    is_folder = False

    @property
    def full_init_file(self) -> str:
        if self.is_folder:
            return get_folder_from_cloud_repo(self.init_file)
        else:
            return get_file_from_cloud_test_repo(self.init_file)

    def test_init(self):
        inst = self.klass(self.full_init_file, **self.init_kwargs)
        self.assertIsInstance(inst, self.klass)


class FromURLTesterMixin(MixinTesterBase):
    url: str
    url_kwargs: dict = {}

    @property
    def full_url(self):
        return r"https://storage.googleapis.com/pylinac_demo_files/" + self.url

    def test_from_url(self):
        self.klass.from_url(self.full_url, **self.url_kwargs)


class FigData(TypedDict):
    title: str
    num_traces: int
    x_label: str
    y_label: str
    has_legend: bool


class PlotlyTestMixin:
    instance: Callable
    num_figs: int
    fig_data: dict[int, FigData] = {}

    def test_plotly_render(self):
        figs = self.instance.plotly_analyzed_images(show=False)
        self.assertEqual(len(figs), self.num_figs)
        keys = list(figs.keys())
        for f_num, f_data in self.fig_data.items():
            fig = figs[keys[f_num]]
            if f_data["num_traces"]:
                self.assertEqual(len(fig.data), f_data["num_traces"])
            self.assertIn(f_data["title"], fig.layout.title.text)
            if f_data.get("has_legend", True):
                self.assertEqual(fig.layout.showlegend, True)
            if f_data.get("x_label"):
                self.assertIn(f_data["x_label"], fig.layout.xaxis.title.text)
            if f_data.get("y_label"):
                self.assertIn(f_data["y_label"], fig.layout.yaxis.title.text)

    def test_plotly_options(self):
        figs = self.instance.plotly_analyzed_images(
            show=False, show_legend=False, show_colorbar=False
        )
        for fig in figs.values():
            # depending on the plot type, showlegend might be False or None
            self.assertNotEqual(fig.layout.showlegend, True)
