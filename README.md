# pie-core

<a href="https://huggingface.co/docs/hub/index"><img alt="Huggingface Hub" src="https://img.shields.io/badge/-Huggingface--Hub-darkslateblue?style=flat&logo=huggingface&labelColor=darkslateblue"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=lightning&logoColor=white"></a>
<a href="https://github.com/ArneBinder/pytorch-ie"><img alt="PyTorch-IE" src="https://img.shields.io/badge/-PyTorch--IE-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![PyPI](https://img.shields.io/pypi/v/pie-core.svg)][pypi status]
[![Tests](https://github.com/arnebinder/pie-core/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/arnebinder/pie-core/branch/main/graph/badge.svg)][codecov]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

This contains the core modules of [PyTorch-IE](https://github.com/ArneBinder/pytorch-ie).

## Setup

```bash
pip install pie-core
```

To install the latest version from GitHub:

```bash
pip install git+https://git@github.com/ArneBinder/pie-core.git
```

## Development

### Setup

1. This project is build with [Poetry](https://python-poetry.org/). See here for [installation instructions](https://python-poetry.org/docs/#installation).
2. Get the code and switch into the project directory:
   ```bash
   git clone https://github.com/ArneBinder/pie-core
   cd pie-core
   ```
3. Create a virtual environment and install the dependencies (including development dependencies):
   ```bash
   poetry install --with dev
   ```

Finally, to run any of the below commands, you need to activate the virtual environment:

```bash
poetry shell
```

Note: You can also run commands in the virtual environment without activating it first: `poetry run <command>`.

### Code Formatting, Linting and Static Type Checking

```bash
pre-commit run -a
```

### Testing

run all tests with coverage:

```bash
pytest --cov --cov-report term-missing
```

### Releasing

1. Create the release branch:
   `git switch --create release main`
2. Increase the version:
   `poetry version <PATCH|MINOR|MAJOR>`,
   e.g. `poetry version patch` for a patch release. If the release contains new features, or breaking changes,
   bump the minor version (this project has no main release yet). If the release contains only bugfixes, bump
   the patch version. See [Semantic Versioning](https://semver.org/) for more information.
3. Commit the changes:
   `git commit --message="release <NEW VERSION>" pyproject.toml`,
   e.g. `git commit --message="release 0.13.0" pyproject.toml`
4. Push the changes to GitHub:
   `git push origin release`
5. Create a PR for that `release` branch on GitHub.
6. Wait until checks passed successfully.
7. Merge the PR into the main branch. This triggers the GitHub Action that creates all relevant release
   artefacts and also uploads them to PyPI.
8. Cleanup: Delete the `release` branch. This is important, because otherwise the next release will fail.

## 📄 License

- **pie-core** is released under the **MIT License** – see [`LICENSE`](LICENSE).
- Third-party licenses and attributions are listed in [`LICENSES/THIRD_PARTY_LICENSES.md`](LICENSES/THIRD_PARTY_LICENSES.md).

[black]: https://github.com/psf/black
[codecov]: https://app.codecov.io/gh/arnebinder/pie-core
[pre-commit]: https://github.com/pre-commit/pre-commit
[pypi status]: https://pypi.org/project/pie-core/
[tests]: https://github.com/arnebinder/pie-core/actions?workflow=Tests
