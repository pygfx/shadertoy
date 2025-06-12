# How to contribute

We welcome all contributions to this project. Be it bug reports or feature requests via the [Issues tab](https://github.com/pygfx/shadertoy/issues) or code contributions via [Pull Requests](https://github.com/pygfx/shadertoy/pulls).


## Submitting a pull request 

**Note**: `git` and basic git knowledge is required.

Find an open issue or task that you wish to work on. Check if there are no PRs already addressing it.

1. Fork the [repository](https://github.com/pygfx/shadertoy) using the "Fork" button.
2. Clone the repository locally and set the upstream remote to the original repository.
```bash
$ git clone https://github.com/<your-github-handle>/shadertoy.git
$ cd shadertoy
$ git remote add upstream https://github.com/pygfx/shadertoy.git
```
3. Create a new branch for your changes. Make sure to branch off from newest `main` branch.
```bash
$ git checkout main
$ git fetch upstream
$ git pull 
$ git checkout -b <your-change>
```
4. Ensure that your environment is set up for development. See [Developers](#developers) for more information
5. Make your changes and commit them with a meaningful commit message.
```bash
$ git add .
$ git commit -m "Add my-change"
```
6. Push your changes to your fork.
```bash
$ git push -u origin <your-change>
```
7. Once your changes are ready, ensure that your changes are formatted and pass tests. See [Testing](#testing) for details.

Go to your fork on GitHub and click the "Compare & pull request" button. Link any connected issues in the description. It is okay to open a draft pull request should your work be not yet complete.

You might be asked to make changes to your pull request. Additional commits you push to your branch will show up on the pull request.

After your pull request is approved, it will be merged into the main branch.

## Developers

* Clone the repo.
* Create a virtual environment using `python -m venv .venv` (Optional)
* Install using `.venv/bin/pip install -e .[dev]`
* Use `.venv/bin/ruff format .` to apply autoformatting.
* Use `.venv/bin/ruff check . --fix` to run the linter.
* Use `.venv/bin/pytest .` to run the tests.
* Use `.venv/bin/pip wheel -w dist --no-deps .` to build a wheel.

*Note*: Replace `/bin/` with `/Scripts/` on Windows.

## Testing

The test suite is divided into multiple parts:

* `pytest -v tests` runs the core unit tests.
* `pytest -v examples` tests the examples.

There are two types of tests for examples included:

### Type 1: Checking if examples can run

When running the test suite, pytest will run every example in a subprocess, to
see if it can run and exit cleanly. You can opt out of this mechanism by
including the comment `# run_example = false` in the module.

### Type 2: Checking if examples output an image

You can also (independently) opt-in to output testing for examples, by including
the comment `# test_example = true` in the module. Output testing means the test
suite will attempt to import the `shader` instance global from your example, and
call it to see if an image is produced.

To support this type of testing, ensure the following requirements are met:

* The `shader` variable is a `Shadertoy` instance exposed as a global in the module.

Reference screenshots are stored in the `examples/screenshots` folder, the test
suite will compare the rendered image with the reference.

Note: this step will be skipped when not running on CI. Since images will have
subtle differences depending on the system on which they are rendered, that
would make the tests unreliable.

For every test that fails on screenshot verification, diffs will be generated
for the rgb and alpha channels and made available in the
`examples/screenshots/diffs` folder. On CI, the `examples/screenshots` folder
will be published as a build artifact so you can download and inspect the
differences.

If you want to update the reference screenshot for a given example, you can grab
those from the build artifacts as well and commit them to your branch.