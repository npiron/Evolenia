# Contributing to EvoLenia

Thank you for your interest in contributing to EvoLenia! This document provides guidelines and information for contributors.

## 🎯 Project Vision

EvoLenia aims to be a scientifically rigorous platform for studying emergent evolution in artificial life systems. Contributions should maintain this focus on:
- **Scientific accuracy** — Implementations should match published algorithms (Lenia, Flow-Lenia, Gray-Scott)
- **Performance** — GPU acceleration is critical; avoid CPU bottlenecks
- **Reproducibility** — Deterministic behavior for scientific experiments
- **Clarity** — Code should be well-documented and understandable

## 🛠️ Development Setup

### Prerequisites
- Rust 1.75+ ([rustup.rs](https://rustup.rs/))
- GPU with Vulkan, Metal, or DX12 support
- Git

### Building from Source
```bash
git clone https://github.com/npiron/Evolenia.git
cd Evolenia
cargo build --release
cargo run --release
```

### Testing
```bash
# Run unit tests (when implemented)
cargo test

# Check for compilation errors
cargo check

# Run with detailed logging
RUST_LOG=debug cargo run --release
```

## 📝 Code Style

- Follow Rust standard style (`cargo fmt`)
- Use `cargo clippy` for linting
- Document all public functions with `///` comments
- WGSL shaders: Include biological/physical interpretation in comments

## 🐛 Bug Reports

When filing an issue, please include:
1. **Hardware**: GPU model, OS version
2. **Reproduction steps**: Minimal example to trigger the bug
3. **Expected vs actual behavior**
4. **Logs**: Run with `RUST_LOG=debug` and attach output

## ✨ Feature Requests

Before suggesting a feature:
1. Check existing issues and [INI.MD](INI.MD) roadmap
2. Explain the **scientific motivation** (not just "it would be cool")
3. Provide references if applicable (papers, algorithms)

## 🔬 Research Contributions

Publishing results from EvoLenia? Please:
- Credit the project: `github.com/npiron/Evolenia`
- Share your findings in Discussions (we'd love to see them!)
- Consider contributing your visualization/analysis code back

## 📚 Areas for Contribution

### High Priority
- [x] **Metrics export** — Automatic CSV logging of genetic entropy, species count
- [x] **State save/load** — Binary snapshot save/load with magic header
- [ ] **WebGPU build** — Port to run in browser via wasm-bindgen
- [x] **Unit tests** — Test conservation laws, genome segregation, PRNG (55 tests)

### Medium Priority
- [ ] **Multi-GPU support** — Distribute computation across GPUs
- [ ] **Phylogenetic tree** — Visualize species lineages over time
- [ ] **Video export** — Render time-lapse videos (ffmpeg integration)
- [x] **Configuration files** — TOML for simulation parameters

### Advanced
- [ ] **3D Lenia** — Extend to volumetric cellular automata
- [ ] **Neural network genomes** — Replace fixed rules with learned behaviors
- [ ] **Coevolution** — Multiple competing species

## 🔍 Code Review Process

Pull requests should:
1. **Pass CI** (once set up: `cargo test`, `cargo fmt --check`, `cargo clippy`)
2. **Include tests** for new features (especially physics/genetics code)
3. **Update documentation** (README, INI.MD, inline comments)
4. **Explain scientific rationale** in the PR description

## 💡 Questions?

- **Technical questions**: Open a GitHub Issue
- **Scientific discussions**: Use GitHub Discussions
- **Contact maintainer**: nicolas.piron@example.com (if public)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License (see [LICENSE](LICENSE)).

---

**Thank you for helping advance artificial life research!** 🌌🦀
