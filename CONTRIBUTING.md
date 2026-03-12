# Contributing to GreenScore

Thank you for your interest in contributing to the **GreenScore Climate-Adjusted CPD Engine**. This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/GreenScore-climate-cpd-engine.git
   cd GreenScore-climate-cpd-engine
   ```

2. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the test suite** to verify your setup:
   ```bash
   python -m pytest tests/ -v
   ```

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure:
   - All existing tests pass
   - New code includes tests where appropriate
   - Code follows existing style conventions

3. Run the full test suite:
   ```bash
   python -m pytest tests/ -v --tb=short
   ```

4. Commit with a descriptive message:
   ```bash
   git commit -m "feat: brief description of changes"
   ```

5. Push and open a Pull Request against `main`.

## Code Style

- Follow PEP 8 conventions
- Maximum line length: 120 characters
- Use type hints for function signatures
- Constants and configuration belong in `config.py`

## Project Structure

| File | Purpose |
|---|---|
| `config.py` | Centralised configuration — all constants live here |
| `cpd_engine.py` | Data loading, feature engineering, XGBoost training |
| `physical_risk.py` | Physical hazard overlay (state → risk score → PD uplift) |
| `transition_risk.py` | NGFS carbon pricing overlay (sector × scenario → PD uplift) |
| `app.py` | Streamlit dashboard |
| `report_gen.py` | RBI disclosure PDF generation |
| `tests/` | Unit tests |

## Adding New Risk Factors

1. Add scores/constants to `config.py`
2. Create or update the risk module (e.g., `physical_risk.py`)
3. Add tests in `tests/test_risk_modules.py`
4. Integrate into `app.py` pipeline

## Reporting Issues

Use [GitHub Issues](https://github.com/Aashik1701/GreenScore-climate-cpd-engine/issues) with:
- Clear title and description
- Steps to reproduce (if applicable)
- Expected vs. actual behavior

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
