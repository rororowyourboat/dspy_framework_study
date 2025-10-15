# Repository Guidelines

## Project Structure & Module Organization
The repository centers on standalone DSPy scripts that can be run directly. Keep executable examples inside `examples/`, mirroring the existing trio of scenario-focused agents; place shared helpers beside them if more than one script imports the code. `main.py` stays a minimal smoke test, while `DSPy_docs.md` captures background notes—extend it only with conceptual references, not runbooks. Treat `requirements.txt` and `pyproject.toml` as the single source of dependency truth, and document any new optional extras you add there.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create and activate a Python 3.13 environment.
- `pip install -r requirements.txt` — install DSPy and pinned runtime dependencies.
- `export OPENAI_API_KEY=... && python examples/basic_showcase.py` — verify API access and the core agent pipeline.
- `python main.py` — run the lightweight sanity check that confirms the environment boots without DSPy calls.

## Coding Style & Naming Conventions
Match the current modules: use four-space indentation, type hints, and triple-quoted docstrings that explain purpose and usage. Keep module and file names in `snake_case.py`, and favor explicit function names that describe the marketing outcome delivered. Follow PEP 8 defaults for imports and line length; if you add formatting tooling (e.g., `ruff` or `black`), check it into `pyproject.toml` alongside rationale.

## Testing Guidelines
There is no automated suite yet, so rely on running the relevant example script end-to-end when you touch it, and capture key console output in the pull request. When a change warrants regression coverage, introduce `pytest` under a new `tests/` directory, naming files `test_<feature>.py` and targeting pure Python units that do not require live model calls. Isolate API-dependent flows behind interfaces so they can be mocked, and record any manual steps in the PR description.

## Commit & Pull Request Guidelines
Commits in this repository use short, capitalized imperatives (e.g., "Add DSPy marketing demo scripts"), so continue that pattern and keep the scope to one topic per commit. Reference related issues in the body, explain why the change matters, and note any follow-up work. Pull requests should summarize the scenario covered, list commands used for validation, and include screenshots or sample console excerpts whenever new prompts or outputs are introduced.

## Security & Configuration Tips
Never hard-code API keys or credentials; rely on environment variables such as `OPENAI_API_KEY` and optionally `DSPY_MODEL`. If you add new providers, document the required secrets in both `README.md` and inline comments where configuration occurs. Review scripts for accidental logging of sensitive payloads before submitting, especially when printing intermediate reasoning traces.
