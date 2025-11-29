# Contributing Guidelines

Thank you for your interest in contributing to the Hotel Booking Cancellation Prediction project! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Expected Behavior

- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community and project

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling, insulting, or derogatory remarks
- Publishing others' private information
- Any conduct that would be inappropriate in a professional setting

---

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report:
1. **Check existing issues** to avoid duplicates
2. **Verify the bug** is reproducible
3. **Collect information** about your environment

**Bug Report Template:**

```markdown
**Description:**
A clear description of the bug

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. ...

**Expected Behavior:**
What you expected to happen

**Actual Behavior:**
What actually happened

**Environment:**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.9.5]
- Package versions: [paste relevant lines from pip list]

**Additional Context:**
Any other relevant information, logs, or screenshots
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

1. **Use a clear title** describing the enhancement
2. **Provide detailed description** of the proposed functionality
3. **Explain why** this enhancement would be useful
4. **Include examples** if applicable

**Enhancement Template:**

```markdown
**Feature Description:**
Clear description of the proposed feature

**Use Case:**
Why this feature would be valuable

**Proposed Implementation:**
How you think this could be implemented (optional)

**Alternatives Considered:**
Other solutions you've considered (optional)
```

### Contributing Code

We welcome code contributions! Here are some areas where you can help:

1. **Bug fixes** - Fix reported issues
2. **New features** - Implement feature requests
3. **Improvements** - Enhance existing functionality
4. **Documentation** - Improve or expand documentation
5. **Tests** - Add or improve test coverage
6. **Performance** - Optimize code performance

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Hotel.git
cd Hotel

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/Hotel.git
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt

# Install development dependencies (if you create a requirements-dev.txt)
pip install -r requirements-dev.txt
```

### 4. Create a Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

---

## Coding Standards

### Python Style Guide

Follow **PEP 8** conventions:

#### Naming Conventions

```python
# Classes: PascalCase
class DataProcessor:
    pass

# Functions and methods: snake_case
def process_data():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3

# Private methods: prefix with underscore
def _internal_method():
    pass
```

#### Code Formatting

```python
# Use 4 spaces for indentation (not tabs)
def example_function(param1, param2):
    if param1:
        return param2
    return None

# Maximum line length: 100 characters
# Break long lines appropriately
result = some_function(
    argument1,
    argument2,
    argument3
)

# Two blank lines between top-level definitions
class FirstClass:
    pass


class SecondClass:
    pass


# One blank line between methods
class MyClass:
    def first_method(self):
        pass
    
    def second_method(self):
        pass
```

### Documentation

#### Docstrings

Use Google-style docstrings:

```python
def process_booking(booking_data, validate=True):
    """
    Process a hotel booking and predict cancellation.
    
    This function takes booking data, validates it if requested,
    and returns a cancellation prediction.
    
    Args:
        booking_data (dict): Dictionary containing booking information
        validate (bool, optional): Whether to validate input. Defaults to True.
        
    Returns:
        dict: Dictionary containing:
            - prediction (str): 'Canceled' or 'Not Canceled'
            - probability (float): Cancellation probability
            - risk_level (str): 'Low', 'Medium', or 'High'
            
    Raises:
        ValueError: If booking_data is missing required fields
        
    Example:
        >>> booking = {'no_of_adults': 2, 'lead_time': 100, ...}
        >>> result = process_booking(booking)
        >>> print(result['prediction'])
        'Not Canceled'
    """
    pass
```

#### Comments

```python
# Use comments to explain WHY, not WHAT
# Good:
# Calculate adjusted price to account for seasonal variations
adjusted_price = base_price * seasonal_factor

# Bad:
# Multiply base_price by seasonal_factor
adjusted_price = base_price * seasonal_factor
```

### Type Hints

Use type hints for better code clarity:

```python
from typing import List, Dict, Optional, Tuple

def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    pass

def get_config(key: str) -> Optional[str]:
    """Get configuration value."""
    pass
```

### Error Handling

```python
# Use specific exceptions
try:
    data = load_data(path)
except FileNotFoundError:
    logger.error(f"Data file not found: {path}")
    raise
except pd.errors.EmptyDataError:
    logger.error(f"Data file is empty: {path}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading data: {str(e)}")
    raise

# Avoid bare except
# Bad:
try:
    risky_operation()
except:  # Don't do this
    pass

# Good:
try:
    risky_operation()
except SpecificException as e:
    handle_error(e)
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning, but program continues")
logger.error("Error occurred, functionality affected")
logger.critical("Critical error, program may not continue")

# Include context in log messages
logger.info(f"Processing {len(data)} records")
logger.error(f"Failed to load model from {path}: {error}")
```

---

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no code change)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

#### Examples

```bash
feat(data_loader): add support for CSV with different encodings

- Added encoding parameter to DataLoader
- Auto-detect encoding if not specified
- Updated documentation

Closes #123

---

fix(preprocess): handle missing values in categorical features

Previously, missing categorical values caused preprocessing to fail.
Now they are filled with the mode of the column.

Fixes #456

---

docs(readme): update installation instructions

- Added troubleshooting section
- Updated requirements
- Added Windows-specific notes
```

### Atomic Commits

- Make small, focused commits
- Each commit should represent a single logical change
- Commit messages should clearly describe the change

```bash
# Good: Separate commits for separate concerns
git commit -m "feat(train): add XGBoost hyperparameter tuning"
git commit -m "docs(api): document new tuning parameters"

# Bad: One large commit for multiple changes
git commit -m "Add features and fix bugs and update docs"
```

---

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest upstream changes

```bash
git checkout main
git pull upstream main
git checkout your-feature-branch
git rebase main
```

2. **Test your changes**

```bash
# Run the full pipeline
python main.py

# Test individual modules
python -m src.data_loader
python -m src.preprocess
python -m src.feature_engineering
python -m src.train

# Run tests (if test suite exists)
pytest tests/
```

3. **Update documentation** if needed
   - Update README.md for new features
   - Update API_REFERENCE.md for API changes
   - Add/update docstrings

4. **Check code quality**

```bash
# Format code (if using black)
black src/

# Check style (if using flake8)
flake8 src/

# Type checking (if using mypy)
mypy src/
```

### Creating the Pull Request

1. **Push your branch**

```bash
git push origin your-feature-branch
```

2. **Create PR on GitHub**

3. **Fill out the PR template:**

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] No new warnings
- [ ] Added tests (if applicable)
- [ ] All tests passing
- [ ] Updated CHANGELOG (if applicable)

## Related Issues
Closes #issue_number
```

### PR Review Process

1. **Maintainers will review** your PR
2. **Address feedback** by making additional commits
3. **Once approved**, maintainers will merge

### After Merge

```bash
# Update your local repository
git checkout main
git pull upstream main

# Delete your feature branch
git branch -d your-feature-branch
```

---

## Issue Guidelines

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** - your question might be answered
3. **Verify the issue** is reproducible

### Creating a Good Issue

- **Use a clear, descriptive title**
- **Provide context** and background
- **Include steps to reproduce** (for bugs)
- **Add logs or screenshots** if relevant
- **Specify your environment**
- **Label appropriately** (bug, enhancement, question)

### Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements to documentation
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - Further information requested
- `wontfix` - This will not be worked on

---

## Development Best Practices

### 1. Keep Changes Focused

- One feature/fix per PR
- Don't mix refactoring with new features
- Keep PRs reasonably sized

### 2. Write Tests

```python
# Example test structure
def test_data_loader():
    """Test DataLoader functionality."""
    loader = DataLoader('test_data.csv')
    data = loader.load_data()
    
    assert data is not None
    assert len(data) > 0
    assert 'booking_status' in data.columns

def test_preprocessing():
    """Test preprocessing pipeline."""
    # Setup
    preprocessor = DataPreprocessor()
    X_train, y_train = get_test_data()
    
    # Execute
    X_processed, y_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Verify
    assert X_processed.shape[0] == X_train.shape[0]
    assert y_processed.shape[0] == y_train.shape[0]
```

### 3. Document Your Code

- Add docstrings to all public functions and classes
- Update README for significant changes
- Include examples in documentation

### 4. Performance Considerations

- Profile code for performance bottlenecks
- Optimize algorithms when possible
- Consider memory usage for large datasets

### 5. Backwards Compatibility

- Avoid breaking changes when possible
- If breaking changes are necessary:
  - Document them clearly
  - Provide migration guide
  - Use deprecation warnings

```python
import warnings

def old_function():
    """Deprecated function."""
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

---

## Questions?

If you have questions about contributing:

1. **Check the documentation** in the `docs/` folder
2. **Open an issue** with the `question` label
3. **Join discussions** in existing issues or PRs

---

## Recognition

Contributors will be recognized in:
- The project README
- Release notes for significant contributions

Thank you for contributing! 🎉
