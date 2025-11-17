# Contributing to sgRNA-TAC

Thanks for your interest in improving sgRNA-TAC! To keep the research artifact reproducible, please follow the guidelines below.

## Getting Started

1. Fork the repository and create a feature branch.
2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
3. Run the test suite (if applicable) before submitting changes.

## Coding Standards

- Place new source files inside `sgrna_tac/`.
- Keep functions focused and write docstrings for public APIs.
- Prefer type hints for new code.

## Pull Requests

- Describe motivation, key changes, and any new dependencies.
- Include usage/testing notes so the research team can reproduce results.
- Ensure linting passes: `python -m compileall sgrna_tac`.

## Questions?

Please open an issue or reach out to the maintainers listed in `README.md`.

