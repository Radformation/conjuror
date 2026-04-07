import nox
from nox import Session


def install_project_groups(session: Session, *groups: str) -> None:
    args = ["sync", "--active", "--locked", "--no-default-groups"]
    for group in groups:
        args.extend(["--group", group])
    session.run_install("uv", *args, external=True)


@nox.session(reuse_venv=True, venv_backend="uv|virtualenv")
def serve_docs(session: Session):
    install_project_groups(session, "docs")
    session.run(
        "sphinx-autobuild",
        "docs/source",
        "docs/build",
        "--port",
        "8717",
        "--open-browser",
    )


@nox.session(reuse_venv=True, venv_backend="uv|virtualenv")
def build_docs(session: Session):
    """Build the docs; used in CI pipelines to test the build. Will always rebuild and will always fail if there are any warnings"""
    install_project_groups(session, "docs")
    session.run(
        "sphinx-build",
        "docs/source",
        "docs/build",
        "-W",
        "--keep-going",
        "-a",
        "-E",
        "-b",
        "html",
        "-q",
    )
