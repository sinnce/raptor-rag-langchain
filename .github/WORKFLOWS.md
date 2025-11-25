# GitHub Actions Workflows

This project uses GitHub Actions for continuous integration and deployment. Below is a description of each workflow.

## 📋 Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Trigger**: Push to `main`/`develop`, Pull Requests

**Jobs**:

#### Lint and Format
- **Python versions**: 3.10, 3.11, 3.12
- **Checks**:
  - `ruff check`: Linting with auto-fix
  - `ruff format`: Code formatting validation
  - `mypy`: Static type checking
- **Artifacts**: None (only pass/fail)

#### Tests
- **Python versions**: 3.10, 3.11, 3.12
- **Steps**:
  - Run `pytest` with coverage
  - Upload coverage reports to Codecov
- **Coverage threshold**: Monitored (currently no required minimum)

#### Pre-commit
- **Purpose**: Run all pre-commit hooks
- **Python version**: Latest (3.10)
- **Uses**: `pre-commit/action@v3.0.0`

#### Security
- **Checks**:
  - `bandit`: Security issue detection
  - `safety`: Dependency vulnerability scanning
- **Note**: These run but don't fail the CI (exit 0)

#### Build
- **Depends on**: Lint, Tests jobs
- **Purpose**: Verify the package can be built
- **Tool**: `python -m build`

### 2. Code Quality Workflow (`.github/workflows/code-quality.yml`)

**Trigger**: Push to `main`/`develop`, Pull Requests

**Jobs**:

#### Quality Analysis
- **Python version**: 3.10
- **Checks**:
  - `pylint`: Extended linting with optional output
  - `coverage report`: Test coverage summary
  - `pip check`: Dependency conflict detection
- **Note**: All checks are optional (`continue-on-error: true`)

## 🚀 Workflow Status

View workflow status on GitHub:
- [CI Status](https://github.com/yourusername/raptor-rag-langchain/actions/workflows/ci.yml)
- [Code Quality Status](https://github.com/yourusername/raptor-rag-langchain/actions/workflows/code-quality.yml)

## 📊 Badges

Add these badges to your README:

```markdown
[![CI](https://github.com/yourusername/raptor-rag-langchain/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/raptor-rag-langchain/actions/workflows/ci.yml)
[![Code Quality](https://github.com/yourusername/raptor-rag-langchain/actions/workflows/code-quality.yml/badge.svg)](https://github.com/yourusername/raptor-rag-langchain/actions/workflows/code-quality.yml)
```

## 🔧 Local Equivalents

Run these commands locally to replicate CI checks:

```bash
# Linting (CI: Lint job)
uv run ruff check src/ tests/

# Formatting check (CI: Lint job)
uv run ruff format --check src/ tests/

# Type checking (CI: Lint job)
uv run mypy src/

# Tests with coverage (CI: Test job)
uv run pytest --cov=src --cov-report=xml

# Pre-commit (CI: Pre-commit job)
pre-commit run --all-files

# Security checks (CI: Security job)
uv run bandit -r src/ -ll
uv run safety check

# Code quality (CI: Quality job)
uv run pylint src/ --exit-zero
```

## ✅ Passing All Checks

To ensure your PR passes all checks:

1. **Before committing:**
   ```bash
   pre-commit run --all-files
   ```

2. **Before pushing:**
   ```bash
   # Run tests
   uv run pytest

   # Check types
   uv run mypy src/

   # Check linting
   uv run ruff check src/
   ```

3. **After pushing:**
   - Check GitHub Actions tab
   - All workflows should be green ✅
   - Code coverage should not decrease

## 🐛 Troubleshooting

### Pre-commit Hook Failures

If a workflow fails on pre-commit checks:

```bash
# Run pre-commit to see issues
pre-commit run --all-files

# Fix issues automatically
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
```

### Test Failures

```bash
# Run tests locally first
uv run pytest -v

# Run specific test
uv run pytest tests/test_module.py::test_function -v
```

### Coverage Decrease

```bash
# Check coverage report
uv run pytest --cov=src --cov-report=term-missing
```

### Type Checking Failures

```bash
# Run mypy with verbose output
uv run mypy src/ --show-error-codes
```

## 📈 Metrics

### Current Status

- **Python Support**: 3.10, 3.11, 3.12
- **Test Framework**: pytest
- **Coverage Tool**: pytest-cov
- **Type Checker**: mypy
- **Linter**: ruff
- **Formatter**: ruff format
- **Security**: bandit, safety
- **Code Complexity**: radon

### Recommended Targets

- Test Coverage: ≥ 80%
- Type Coverage: 100% (strict mode enabled)
- Linting: 0 warnings
- Security: 0 critical issues

## 🔐 Secrets & Configuration

### Required Secrets

Add these to your GitHub repository settings (`Settings → Secrets`):

- `CODECOV_TOKEN`: For uploading coverage to Codecov (optional)

### Environment Setup

All workflows use Python's default installation. For API key testing:

- Create test fixtures that mock external APIs
- Use environment variables (stored as secrets if needed)
- Never commit real API keys

## 📝 Customization

To modify workflows:

1. Edit `.github/workflows/ci.yml` or `.github/workflows/code-quality.yml`
2. Commit and push to a branch
3. Test the workflow by creating a PR
4. Merge once verified

## 📚 References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
