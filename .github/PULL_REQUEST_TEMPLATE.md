## What does this change?

<!-- One or two sentences. Link the issue it closes: "Closes #123". -->

## Example

<!-- A snippet that works now and didn't before (or now behaves differently). -->

```text

```

## Ambiguity check

<!-- If this adds or changes syntax: could it mean something else in another
     supported language? Which reading did you choose, and why? Write "n/a"
     for changes that don't touch the parser. -->

## Checklist

- [ ] `cargo fmt --all --check` passes
- [ ] `cargo clippy --all-targets -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] Added or updated a golden test (`tests/golden/`) or unit test
- [ ] Updated `docs/LANGUAGE.md` if this resolves a syntax ambiguity
- [ ] Added a line to `CHANGELOG.md` under **Unreleased**
