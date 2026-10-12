# Security policy

## What Chimera is (and isn't)

Chimera is a general-purpose interpreter, like `python` or `node`. A Chimera
program can read and write files, read environment variables, and (with
the Claude backend) make network requests. **Running a Chimera program
means trusting it, just as you would a Python script.** Chimera is not a
sandbox, and running untrusted code with it isn't a supported use.

The same applies to `please` and `ai.code` with the Claude backend: they run
code that a model wrote, in the current scope. Use the offline or mock
backend when that isn't what you want.

## What counts as a vulnerability

Please report:

- ways the Claude backend could leak `ANTHROPIC_API_KEY`. For example,
  exposing it on a command line, in process listings, in error messages or
  in logs. The key is passed to `curl` on stdin by design.
- crashes of the interpreter process itself (a Rust panic or a stack
  overflow) triggered by a source file. These are robustness bugs, but we
  treat them seriously.
- anything in the CLI or the build that runs code or touches files the user
  didn't ask for.

## Reporting

Please **don't open a public issue** for a security problem. Use GitHub's
private vulnerability reporting instead: the repository's **Security** tab,
then **Report a vulnerability**. Include a minimal program or steps that
reproduce the problem.

We aim to acknowledge reports within a week. Once a fix is released, we'll
credit you in the changelog unless you'd rather stay anonymous.

## Supported versions

Only the latest commit on the default branch is supported. Chimera hasn't
reached 1.0 yet.
