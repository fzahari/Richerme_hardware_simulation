# Contributing to Richerme Quantum Hardware

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Richerme Quantum Hardware library.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Richerme_hardware_simulation.git
   cd Richerme_hardware_simulation
   ```
3. **Set up the development environment**:
   ```bash
   conda create -n qiskit-fresh python=3.9 -y
   conda activate qiskit-fresh
   pip install -r requirements.txt
   pip install -e .
   ```

## Development Guidelines

### Code Standards

This project follows **SOLID principles** and **Clean Code practices**. Please review [DEVELOPMENT.md](DEVELOPMENT.md) for comprehensive coding standards.

Key principles:
- **Small functions** (< 20 lines preferred)
- **Descriptive names** (no unclear abbreviations)
- **Type hints** on all public functions
- **Docstrings** with examples for all public APIs
- **No side effects** (pure functions preferred)
- **≤ 3 function arguments** (use dataclasses for more)

### Code Style

- Follow PEP 8 style guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use meaningful variable names:
  - `n`, `N` for number of qubits/ions
  - `i`, `j`, `k` for indices
  - `U`, `V` for unitary matrices
  - `H` for Hamiltonians
  - `J` for coupling matrices
  - `B` for mode matrices

### Testing Requirements

**All contributions must include tests.**

```bash
# Run tests before submitting
pytest tests/ -v

# Your changes should not break existing tests
# Add new tests for new functionality
```

Test requirements:
- **Unit tests** for individual functions
- **Integration tests** for complete workflows
- **Numerical precision** validation (≤ 1e-12 for gate synthesis)
- **One concept per test** (focused, clear tests)

Example test structure:
```python
def test_new_feature():
    """Test that new feature works correctly."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = new_feature(input_data)

    # Assert
    assert result.shape == expected_shape
    assert np.allclose(result, expected_result, atol=1e-12)
```

### Documentation Requirements

1. **Docstrings** for all public functions:
   ```python
   def function_name(arg1: type, arg2: type) -> return_type:
       """
       Brief one-line description.

       Longer description explaining purpose, algorithm, and usage.

       Args:
           arg1: Description of arg1
           arg2: Description of arg2

       Returns:
           Description of return value

       Example:
           >>> result = function_name(value1, value2)
           >>> print(result)
           expected_output

       References:
           Paper Name, Equation X
       """
   ```

2. **Comments** should explain WHY, not WHAT:
   ```python
   # GOOD: Explain reasoning
   # Use eigendecomposition for numerical stability
   w, v = np.linalg.eigh(H)

   # BAD: Redundant with code
   # Calculate eigenvalues and eigenvectors
   w, v = np.linalg.eigh(H)
   ```

3. **Update README.md** if adding major features

## Contribution Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-xyz-gate`
- `fix/precision-error`
- `docs/update-examples`

### 2. Make Your Changes

- Write clear, focused commits
- Follow the code standards above
- Add tests for new functionality
- Update documentation

### 3. Test Your Changes

```bash
# Run full test suite
pytest tests/ -v

# Run specific tests
pytest tests/test_richerme_ion_analog.py -v

# Check that examples still work
python examples/demo_zxx.py
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add XYZ gate synthesis method

- Implement n_body_string_xyz() for arbitrary XYZ patterns
- Add unit tests with numerical precision validation
- Update documentation with usage examples
- Achieves 1e-15 fidelity error (machine precision)"
```

Commit message format:
- **First line**: Brief summary (< 50 chars)
- **Blank line**
- **Body**: Detailed description (what and why)

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- **Clear title** describing the change
- **Description** explaining:
  - What was changed and why
  - How it was tested
  - Any breaking changes
  - Related issues (if applicable)

## Pull Request Guidelines

### PR Checklist

Before submitting, ensure:
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] New functionality has tests
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No merge conflicts with main branch

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Types of Contributions

### Bug Fixes

- Include a test that reproduces the bug
- Explain the root cause in the PR description
- Verify the fix with numerical tests

### New Features

- Discuss major features in an issue first
- Follow existing architecture patterns
- Include comprehensive tests and documentation
- Add examples demonstrating usage

### Documentation

- Fix typos, clarify explanations
- Add examples for existing features
- Improve API documentation
- Update outdated information

### Performance Improvements

- Include benchmarks showing improvement
- Ensure numerical accuracy is maintained
- Document any trade-offs

## Questions?

- Open an issue for discussion
- Tag maintainers: @fzahari

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Welcome newcomers and help them learn
- Follow scientific integrity standards

## Scientific Contributions

When contributing algorithms:
- **Cite sources** in docstrings and comments
- **Reference equations** from papers
- **Validate accuracy** against published results
- **Document assumptions** and limitations

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to quantum simulation research!
