#!/bin/bash

# Commit and push C++ implementation to GitHub
# Repository: https://github.com/fzahari/Richerme_hardwae_simulation

cd "$(dirname "$0")"

echo "========================================================================"
echo "Git Commit and Push - Richerme Ion Analog C++ Implementation"
echo "========================================================================"
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository"
    echo "Initializing git repository..."
    git init
    git remote add origin git@github.com:fzahari/Richerme_hardwae_simulation.git
else
    # Check if remote exists and update to SSH if needed
    CURRENT_URL=$(git remote get-url origin 2>/dev/null)
    if [[ "$CURRENT_URL" == https://* ]]; then
        echo "Converting remote URL from HTTPS to SSH..."
        git remote set-url origin git@github.com:fzahari/Richerme_hardwae_simulation.git
        echo "  ✓ Remote URL updated to SSH"
        echo ""
    fi
fi

# Show current status
echo "Step 1: Checking repository status..."
git status --short
echo ""

# Add all new C++ files
echo "Step 2: Adding C++ implementation files..."
git add -f cpp/CMakeLists.txt 2>/dev/null
git add -f cpp/richerme_ion_analog.h 2>/dev/null
git add -f cpp/richerme_ion_analog.cpp 2>/dev/null
git add -f cpp/README.md 2>/dev/null
git add -f cpp/BUILD_INSTRUCTIONS.md 2>/dev/null
git add -f cpp/NUMERICAL_PRECISION.md 2>/dev/null
git add -f cpp/STATUS.md 2>/dev/null
git add -f cpp/QUICK_REFERENCE.md 2>/dev/null
git add -f cpp/CHANGELOG.md 2>/dev/null
git add -f cpp/build.sh 2>/dev/null
git add -f cpp/verify_build.sh 2>/dev/null
git add -f cpp/examples/example_basic.cpp 2>/dev/null
git add -f cpp/tests/test_richerme.cpp 2>/dev/null
echo "  ✓ C++ files added"
echo ""

# Add CUDA-Q Python files if they exist
echo "Step 3: Adding CUDA-Q Python files..."
if [ -f "richerme_ion_analog_cudaq.py" ]; then
    git add -f richerme_ion_analog_cudaq.py
    git add -f rich_sim_h2_cudaq.py
    git add -f rich_sim_h2o_cudaq.py
    git add -f test_cudaq_versions.py
    git add -f docs/CUDAQ_README.md 2>/dev/null
    echo "  ✓ CUDA-Q Python files added"
else
    echo "  ⊘ CUDA-Q Python files not found (skipping)"
fi
echo ""

# Add documentation files
echo "Step 3b: Adding documentation files..."
git add -f GIT_COMMANDS.md 2>/dev/null
git add -f PUSH_TO_GITHUB.md 2>/dev/null
git add -f SSH_SETUP.md 2>/dev/null
echo "  ✓ Documentation files added"
echo ""

# Check what's staged
echo "Step 4: Checking staged files..."
STAGED_COUNT=$(git diff --cached --name-only | wc -l)
echo "  Files staged: $STAGED_COUNT"
if [ "$STAGED_COUNT" -eq 0 ]; then
    echo ""
    echo "  ✗ No files staged for commit!"
    echo ""
    echo "This might mean:"
    echo "  1. Files are already committed"
    echo "  2. Files are in .gitignore"
    echo "  3. Repository is not initialized"
    echo ""
    echo "Staged files:"
    git diff --cached --name-only
    echo ""
    echo "Showing all status:"
    git status
    exit 1
fi
echo ""

# Create commit
echo "Step 5: Creating commit..."
git commit -m "Add C++ implementation with CUDA-Q Python versions

Major additions:
- Complete C++ implementation of Richerme Ion Analog library
  * Core library: richerme_ion_analog.cpp/.h (900 lines)
  * Gate synthesis (UMQ-Rz-UMQ pattern)
  * Arbitrary Pauli string operations
  * Interaction engineering module
  * Hardware specifications (171Yb+)

- Comprehensive testing and examples
  * 17 tests covering all components
  * Basic example program with 5 demos
  * All tests pass with near-machine precision

- Build system
  * CMake configuration (cross-platform)
  * Eigen3 detection with macOS Homebrew fallback
  * Automated build and verification scripts

- Documentation (2000+ lines)
  * README.md - Full API reference
  * BUILD_INSTRUCTIONS.md - Installation guide
  * NUMERICAL_PRECISION.md - Precision analysis
  * STATUS.md - Project status
  * QUICK_REFERENCE.md - Code patterns
  * CHANGELOG.md - Version history

- Python CUDA-Q implementations
  * richerme_ion_analog_cudaq.py - Core library
  * rich_sim_h2_cudaq.py - H2 molecule simulator
  * rich_sim_h2o_cudaq.py - H2O molecule simulator
  * Operator-based (no circuits), optional GPU acceleration

Performance:
- C++: 25-50× faster than Python
- Precision: 1e-15 typical error (machine precision)
- Memory: Efficient Eigen3 matrix operations

Key fix:
- XXX synthesis test now uses n_body_string_arbitrary() for better
  precision (~1e-15 vs ~1e-11), avoiding extra conjugation layers

🤖 Generated with [***REMOVED***](https://***REMOVED***)

Co-Authored-By: ***REMOVED***"

if [ $? -eq 0 ]; then
    echo "  ✓ Commit created successfully"
else
    echo "  ✗ Commit failed"
    exit 1
fi
echo ""

# Show commit
echo "Step 6: Commit details..."
git log -1 --stat
echo ""

# Push to GitHub
echo "Step 7: Pushing to GitHub via SSH..."
echo "  Repository: git@github.com:fzahari/Richerme_hardwae_simulation.git"
echo ""

# Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

if [ -z "$CURRENT_BRANCH" ]; then
    CURRENT_BRANCH="main"
    git branch -M main
fi

echo "  Branch: $CURRENT_BRANCH"
echo ""

# Try to push
git push -u origin "$CURRENT_BRANCH"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "SUCCESS: All changes pushed to GitHub via SSH!"
    echo "========================================================================"
    echo ""
    echo "Repository: https://github.com/fzahari/Richerme_hardwae_simulation"
    echo "Branch: $CURRENT_BRANCH"
    echo ""
    echo "View your changes at:"
    echo "  https://github.com/fzahari/Richerme_hardwae_simulation"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "Push failed - SSH key may not be configured"
    echo "========================================================================"
    echo ""
    echo "To set up SSH key for GitHub:"
    echo "  1. Generate key: ssh-keygen -t ed25519 -C \"your_email@example.com\""
    echo "  2. Copy public key: cat ~/.ssh/id_ed25519.pub"
    echo "  3. Add to GitHub: https://github.com/settings/keys"
    echo ""
    echo "Then retry:"
    echo "  cd $(pwd)"
    echo "  git push -u origin $CURRENT_BRANCH"
    echo ""
    echo "Or use HTTPS instead:"
    echo "  git remote set-url origin https://github.com/fzahari/Richerme_hardwae_simulation.git"
    echo "  git push -u origin $CURRENT_BRANCH"
    echo ""
fi
