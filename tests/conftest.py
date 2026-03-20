"""Shared fixtures for verification tests."""

import os
import json

import pytest
from safetensors import safe_open

from my_sd15.loader import SINGLE_FILE_PATH, load_from_single_file


TESTDATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testdata")


def load_testdata(name):
    """Load all tensors from a testdata safetensors file as torch tensors."""
    path = os.path.join(TESTDATA_DIR, f"{name}.safetensors")
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def single_file_available():
    return os.path.exists(SINGLE_FILE_PATH)


@pytest.fixture(scope="session")
def models():
    clip, unet, vae = load_from_single_file()
    return clip, unet, vae


@pytest.fixture(scope="session")
def metadata():
    path = os.path.join(TESTDATA_DIR, "metadata.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def tokenizer_data():
    return load_testdata("tokenizer")


@pytest.fixture(scope="session")
def clip_data():
    return load_testdata("clip")


@pytest.fixture(scope="session")
def scheduler_data():
    return load_testdata("scheduler")


@pytest.fixture(scope="session")
def vae_data():
    return load_testdata("vae")


@pytest.fixture(scope="session")
def unet_data():
    return load_testdata("unet")


@pytest.fixture(scope="session")
def pipeline_data():
    return load_testdata("pipeline")
