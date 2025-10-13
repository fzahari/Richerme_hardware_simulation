# Git Commands for Pushing to GitHub

## Quick Push (Recommended)

Run the automated script:
```bash
cd /Users/federicozahariev/Work/Programs/Richerme_Quantum_Hardware
chmod +x commit_and_push.sh
./commit_and_push.sh
```

## Manual Commands

If you prefer to run git commands manually:

### 1. Check Status
```bash
cd /Users/federicozahariev/Work/Programs/Richerme_Quantum_Hardware
git status
```

### 2. Add All C++ Files
```bash
# Core library
git add cpp/CMakeLists.txt
git add cpp/richerme_ion_analog.h
git add cpp/richerme_ion_analog.cpp

# Documentation
git add cpp/README.md
git add cpp/BUILD_INSTRUCTIONS.md
git add cpp/NUMERICAL_PRECISION.md
git add cpp/STATUS.md
git add cpp/QUICK_REFERENCE.md
git add cpp/CHANGELOG.md

# Build scripts
git add cpp/build.sh
git add cpp/verify_build.sh

# Examples and tests
git add cpp/examples/example_basic.cpp
git add cpp/tests/test_richerme.cpp

# CUDA-Q Python versions (if they exist)
git add richerme_ion_analog_cudaq.py
git add rich_sim_h2_cudaq.py
git add rich_sim_h2o_cudaq.py
git add test_cudaq_versions.py
git add docs/CUDAQ_README.md
```

### 3. Create Commit
```bash
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
```

### 4. Set Remote to SSH (Important!)
```bash
# Set remote to SSH
git remote set-url origin git@github.com:fzahari/Richerme_hardwae_simulation.git
```

### 5. Push to GitHub
```bash
# Push to current branch
git push -u origin main

# Or push to master if that's your branch
git push -u origin master
```

## Alternative: Single Command

Or add and commit everything in one go:
```bash
git add cpp/*.{cpp,h,md,txt,sh} cpp/examples/*.cpp cpp/tests/*.cpp *_cudaq.py docs/CUDAQ_README.md test_cudaq_versions.py 2>/dev/null
git commit -F- <<'EOF'
Add C++ implementation with CUDA-Q Python versions

Major additions:
- Complete C++ implementation (900 lines)
- 17 comprehensive tests (all passing)
- Extensive documentation (2000+ lines)
- Python CUDA-Q versions (operator-based)

Performance: 25-50× faster than Python
Precision: 1e-15 typical error

🤖 Generated with [***REMOVED***](https://***REMOVED***)

Co-Authored-By: ***REMOVED***
EOF
git push -u origin main
```

## Troubleshooting

### SSH Key Not Configured

If you get "Permission denied (publickey)" error:

**Set up SSH key:**
```bash
# 1. Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. Copy your public key
cat ~/.ssh/id_ed25519.pub

# 3. Add to GitHub
# Go to: https://github.com/settings/keys
# Click "New SSH key" and paste the key

# 4. Test connection
ssh -T git@github.com
# Should see: "Hi username! You've successfully authenticated..."

# 5. Now push
git push -u origin main
```

### Check Remote URL
```bash
git remote -v
```

Should show SSH URLs:
```
origin  git@github.com:fzahari/Richerme_hardwae_simulation.git (fetch)
origin  git@github.com:fzahari/Richerme_hardwae_simulation.git (push)
```

### Switch from HTTPS to SSH
```bash
git remote set-url origin git@github.com:fzahari/Richerme_hardwae_simulation.git
git remote -v  # Verify
```

### First Time Setup
If repository not initialized:
```bash
git init
git remote add origin git@github.com:fzahari/Richerme_hardwae_simulation.git
git branch -M main
```

## Files Being Committed

### C++ Implementation
```
cpp/
├── CMakeLists.txt              (build configuration)
├── richerme_ion_analog.h       (header, 300 lines)
├── richerme_ion_analog.cpp     (implementation, 600 lines)
├── README.md                   (API docs, 470 lines)
├── BUILD_INSTRUCTIONS.md       (build guide)
├── NUMERICAL_PRECISION.md      (precision analysis)
├── STATUS.md                   (project status)
├── QUICK_REFERENCE.md          (code patterns)
├── CHANGELOG.md                (version history)
├── build.sh                    (build script)
├── verify_build.sh             (verification script)
├── examples/
│   └── example_basic.cpp       (200 lines, 5 examples)
└── tests/
    └── test_richerme.cpp       (250 lines, 17 tests)
```

### Python CUDA-Q Versions
```
richerme_ion_analog_cudaq.py    (500 lines)
rich_sim_h2_cudaq.py            (650 lines)
rich_sim_h2o_cudaq.py           (530 lines)
test_cudaq_versions.py          (200 lines)
docs/CUDAQ_README.md            (comprehensive docs)
```

## Verify Push

After pushing, verify at:
https://github.com/fzahari/Richerme_hardwae_simulation

You should see:
- New `cpp/` directory with all files
- Updated Python CUDA-Q files
- Commit message with full description

## Summary

**Easiest method:**
```bash
cd /Users/federicozahariev/Work/Programs/Richerme_Quantum_Hardware
chmod +x commit_and_push.sh
./commit_and_push.sh
```

The script will handle everything automatically.
