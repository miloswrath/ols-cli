# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Rust crate; `main.rs` wires CLI execution, `cli.rs` defines arguments, `config.rs` performs validation, and `regression/` contains preprocess, solver, and reporting modules.
- `data/` provides curated CSV fixtures (`boston_housing`, `energy_efficiency`, `marketing_mix`) for manual smoke tests.
- `target/` is Cargo's build output and should be ignored in commits.

## Build, Test, and Development Commands
- `cargo fmt` formats the codebase using the repo's default `rustfmt` profile.
- `cargo check` performs a fast type-check without generating binaries; run it before committing.
- `cargo test` executes unit and integration tests (add cases as solver logic matures).
- `cargo run -- fit …` runs the CLI; reuse the sample commands in `README.md` when iterating on UX.

## Coding Style & Naming Conventions
- Adhere to Rust 2021 idioms: modules and files in `snake_case`, types and enums in `PascalCase`, functions and variables in `snake_case`.
- Leave explanatory comments only where logic is non-obvious; prefer self-documenting code.
- Keep warnings at zero by fixing `cargo check` output; enable `cargo clippy` locally when touching solver math.

## Testing Guidelines
- Co-locate unit tests in their source modules with `#[cfg(test)]` blocks; name functions `mod_name::tests::case_description`.
- For regression fixtures, add integration tests under `tests/` that run the CLI against the CSVs and assert on report metadata.
- Favor deterministic sample data; update `data/` fixtures instead of random generators.

## Commit & Pull Request Guidelines
- Write imperative commit subjects (e.g., "Add ridge sample dataset") followed by concise bodies when context is needed.
- Reference related issues in the body using `Fixes #123` or `Refs #123`.
- Pull requests should summarize behavior changes, list new commands or flags, and note any manual verification (sample command runs, screenshots of reports).

## Sample Data Workflow Tips
- When adding new fixtures, place them under `data/<scenario>/` and update both `README.md` and inline docs with usage commands.
- Keep datasets under ~1 KB to sustain quick CLI iterations and repository hygiene.
