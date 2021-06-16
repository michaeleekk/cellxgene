import errno
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from os import path
from subprocess import Popen

import click
import pandas as pd
import requests

from backend.czi_hosted.cli.launch import CliLaunchServer
from backend.czi_hosted.common.annotations.hosted_tiledb import AnnotationsHostedTileDB
from backend.czi_hosted.common.annotations.local_file_csv import AnnotationsLocalFile
from backend.czi_hosted.common.config import DEFAULT_SERVER_PORT
from backend.czi_hosted.common.config.app_config import AppConfig
from backend.common.utils.data_locator import DataLocator
from backend.common.errors import ConfigurationError, DatasetAccessError
from backend.common.utils.utils import find_available_port
from backend.common.fbs.matrix import encode_matrix_fbs
from backend.czi_hosted.data_common.matrix_loader import MatrixDataType, MatrixDataLoader
from backend.czi_hosted.db.db_utils import DbUtils
from backend.test import PROJECT_ROOT, FIXTURES_ROOT


def data_with_tmp_tiledb_annotations(ext: MatrixDataType):
    tmp_dir = tempfile.mkdtemp()
    fname = {
        MatrixDataType.H5AD: f"{PROJECT_ROOT}/example-dataset/pbmc3k.h5ad",
        MatrixDataType.CXG: "test/fixtures/pbmc3k.cxg",
    }[ext]
    data_locator = DataLocator(fname)
    config = AppConfig()
    config.update_server_config(
        app__flask_secret_key="secret",
        multi_dataset__dataroot=data_locator.path,
        authentication__type="test",
        authentication__insecure_test_environment=True,
    )
    config.update_default_dataset_config(
        embeddings__names=["umap"],
        presentation__max_categories=100,
        diffexp__lfc_cutoff=0.01,
        user_annotations__type="hosted_tiledb_array",
        user_annotations__hosted_tiledb_array__db_uri="postgresql://postgres:test_pw@localhost:5432",
        user_annotations__hosted_tiledb_array__hosted_file_directory=tmp_dir,
    )

    config.complete_config()

    data = MatrixDataLoader(data_locator.abspath()).open(config)
    annotations = AnnotationsHostedTileDB(
        {
            "user-annotations": True,
            "genesets-save": False,
        },
        tmp_dir,
        DbUtils("postgresql://postgres:test_pw@localhost:5432"),
    )
    return data, tmp_dir, annotations


def data_with_tmp_annotations(ext: MatrixDataType, annotations_fixture=False):
    tmp_dir = tempfile.mkdtemp()
    annotations_file = path.join(tmp_dir, "test_annotations.csv")
    if annotations_fixture:
        shutil.copyfile(f"{FIXTURES_ROOT}/pbmc3k-annotations.csv", annotations_file)
    fname = {
        MatrixDataType.H5AD: f"{PROJECT_ROOT}/example-dataset/pbmc3k.h5ad",
        MatrixDataType.CXG: f"{FIXTURES_ROOT}/pbmc3k.cxg",
    }[ext]
    data_locator = DataLocator(fname)
    config = AppConfig()
    config.update_server_config(
        app__flask_secret_key="secret",
        single_dataset__obs_names=None,
        single_dataset__var_names=None,
        single_dataset__datapath=data_locator.path,
    )
    config.update_default_dataset_config(
        embeddings__names=["umap"],
        presentation__max_categories=100,
        diffexp__lfc_cutoff=0.01,
    )

    config.complete_config()
    data = MatrixDataLoader(data_locator.abspath()).open(config)
    annotations = AnnotationsLocalFile(
        {
            "user-annotations": True,
            "genesets-save": False,
        },
        None,
        annotations_file,
    )
    return data, tmp_dir, annotations


def make_fbs(data):
    df = pd.DataFrame(data)
    return encode_matrix_fbs(matrix=df, row_idx=None, col_idx=df.columns)


def skip_if(condition, reason: str):
    def decorator(f):
        def wraps(self, *args, **kwargs):
            if condition(self):
                self.skipTest(reason)
            else:
                f(self, *args, **kwargs)

        return wraps

    return decorator


def app_config(data_locator, backed=False, extra_server_config={}, extra_dataset_config={}):
    config = AppConfig()
    config.update_server_config(
        app__flask_secret_key="secret",
        single_dataset__obs_names=None,
        single_dataset__var_names=None,
        adaptor__anndata_adaptor__backed=backed,
        single_dataset__datapath=data_locator,
        limits__diffexp_cellcount_max=None,
        limits__column_request_max=None,
    )
    config.update_default_dataset_config(
        embeddings__names=["umap", "tsne", "pca"], presentation__max_categories=100, diffexp__lfc_cutoff=0.01
    )
    config.update_server_config(**extra_server_config)
    config.update_default_dataset_config(**extra_dataset_config)
    config.complete_config()
    return config


def start_test_server(command_line_args=[], app_config_info=None, env=None):
    """
    Command line arguments can be passed in, as well as an app_config.
    This function is meant to be used like this, for example:

    with unit(...) as server:
        r = requests.get(f"{server}/...")
        // check r

    where the server can be accessed within the context, and is terminated when
    the context is exited.
    The port is automatically set using find_available_port, unless passed in as a command line arg.
    The verbose flag is automatically set to True.
    If an app_config is provided, then this function writes a temporary
    yaml config file, which this server will read and parse.
    """

    # if bool(os.environ.get("DEV_INSTALL", False)):
    #     if not bool(os.environ.get("DEV_INSTALLED", False)):
    #         subprocess.run(["pip", "install", "-e", "../../"])
    #         os.environ["DEV_INSTALLED"] = "True"
    #         import pdb
    #         pdb.set_trace()
    # command = ["cellxgene", "--no-upgrade-check", "launch", "--verbose"]
    # if "-p" in command_line_args:
    #     port = int(command_line_args[command_line_args.index("-p") + 1])
    # elif "--port" in command_line_args:
    #     port = int(command_line_args[command_line_args.index("--port") + 1])
    # else:
    #     start = random.randint(DEFAULT_SERVER_PORT, 2 ** 16 - 1)
    #     port = int(os.environ.get("CXG_SERVER_PORT", start))
    #     port = find_available_port("localhost", port)
    #     command += ["--port=%d" % port]
    #
    # command += command_line_args
    #
    # tempdir = None
    # if app_config:
    #     tempdir = tempfile.TemporaryDirectory()
    #     config_file = os.path.join(tempdir.name, "config.yaml")
    #     app_config.write_config(config_file)
    #     command.extend(["-c", config_file])
    #
    # server = f"http://localhost:{port}"
    # ps = Popen(command, env=env)

    if "-p" in command_line_args:
        port = int(command_line_args[command_line_args.index("-p") + 1])
    elif "--port" in command_line_args:
        port = int(command_line_args[command_line_args.index("--port") + 1])
    else:
        start = random.randint(DEFAULT_SERVER_PORT, 2 ** 16 - 1)
        port = int(os.environ.get("CXG_SERVER_PORT", start))
        port = find_available_port("localhost", port)

    app_config = AppConfig()
    server_config = app_config.server_config
    app_config.update_server_config(
        app__verbose=True,
        app__debug=True,
        app__port=port,
        app__host="localhost",
        app__open_browser=False
    )
    try:
        if app_config_info:
            app_config.update_from_config_file(app_config_info)


        # Determine which config options were give on the command line.
        # Those will override the ones provided in the config file (if provided).
        # cli_config = AppConfig()
        # cli_config.update_server_config(
        #     # app__verbose=verbose,
        #     # app__debug=debug,
        #     # app__host=host,
        #     # app__port=port,
        #     # app__open_browser=open_browser,
        #     single_dataset__datapath=datapath,
        #     single_dataset__title=title,
        #     single_dataset__about=about,
        #     single_dataset__obs_names=obs_names,
        #     single_dataset__var_names=var_names,
        #     multi_dataset__dataroot=dataroot,
        #     adaptor__anndata_adaptor__backed=backed,
        # )
        # cli_config.update_default_dataset_config(
        #     app__scripts=scripts,
        #     user_annotations__enable=not disable_annotations,
        #     user_annotations__local_file_csv__file=annotations_file,
        #     user_annotations__local_file_csv__directory=annotations_dir,
        #     user_annotations__ontology__enable=experimental_annotations_ontology,
        #     user_annotations__ontology__obo_location=experimental_annotations_ontology_obo,
        #     presentation__max_categories=max_category_items,
        #     presentation__custom_colors=not disable_custom_colors,
        #     embeddings__names=embedding,
        #     embeddings__enable_reembedding=experimental_enable_reembedding,
        #     diffexp__enable=not disable_diffexp,
        #     diffexp__lfc_cutoff=diffexp_lfc_cutoff,
        # )
        #
        # diff = cli_config.server_config.changes_from_default()
        # changes = {key: val for key, val, _ in diff}
        # app_config.update_server_config(**changes)
        #
        # diff = cli_config.default_dataset_config.changes_from_default()
        # changes = {key: val for key, val, _ in diff}
        # app_config.update_default_dataset_config(**changes)

        # process the configuration
        #  any errors will be thrown as an exception.
        #  any info messages will be passed to the messagefn function.

        def messagefn(message):
            click.echo("[cellxgene] " + message)

        # Use a default secret if one is not provided
        if not server_config.app__flask_secret_key:
            app_config.update_server_config(app__flask_secret_key="SparkleAndShine")

        app_config.complete_config(messagefn)

    except (ConfigurationError, DatasetAccessError) as e:
        raise click.ClickException(e)


    # create the server
    server = CliLaunchServer(app_config)

    if not server_config.app__verbose:
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

    cellxgene_url = f"http://{app_config.server_config.app__host}:{app_config.server_config.app__port}"
    click.echo(f"[cellxgene] Launching! Please go to {cellxgene_url} in your browser.")

    click.echo("[cellxgene] Type CTRL-C at any time to exit.")

    if not server_config.app__verbose:
        f = open(os.devnull, "w")
        sys.stdout = f

    try:
        server.app.run(
            host=server_config.app__host,
            debug=server_config.app__debug,
            port=server_config.app__port,
            threaded=not server_config.app__debug,
            use_debugger=False,
            use_reloader=False,
        )
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            raise click.ClickException("Port is in use, please specify an open port using the --port flag.") from e
        raise

    for _ in range(10):
        try:
            requests.get(f"{server}/health")
            break
        except requests.exceptions.ConnectionError:
            time.sleep(1)

    # if tempfile.tempdir:
    #     tempdir.cleanup()

    return ps, server


def stop_test_server(ps):
    try:
        ps.terminate()
    except ProcessLookupError:
        pass


@contextmanager
def test_server(command_line_args=[], app_config=None, env=None):
    """A context to run the cellxgene server."""

    ps, server = start_test_server(command_line_args, app_config, env)
    try:
        yield server
    finally:
        try:
            stop_test_server(ps)
        except ProcessLookupError:
            pass
