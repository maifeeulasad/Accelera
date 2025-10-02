# Development Guidelines

## Philosophy

We believe that thoughtful, disciplined, and transparent engineering practices lead to long-term team and product success. Our approach balances **incremental value delivery** with **sustainable code quality**, **customer obsession**, and **engineering excellence**.

## Core Beliefs

* **Incremental progress over big bangs** - Small changes that compile and pass tests
* **Learning from existing code** - Study and plan before implementing
* **Pragmatic over dogmatic** - Adapt to project reality
* **Clear intent over clever code** - Be boring but obvious

**(Inspired by Microsoft, Amazon):**

* **Customer-focused design** - Always consider the customer experience when designing APIs, features, and workflows
* **Secure by default** - Build systems that avoid common vulnerabilities by design (e.g., SQL injection, XSS, improper authentication)
* **Operational excellence matters** - Code is only done when it’s monitored, observable, and behaves well in production
* **Two-way door vs one-way door decisions** - Favor quick decisions for reversible choices (Amazon's "bias for action")

---

## Simplicity Means

* Single responsibility per function/class
* Avoid premature abstractions
* No clever tricks - choose the boring solution
* If you need to explain it, it's too complex
* Documenting *what* something does is not necessary, because it's self evident from naming
* Micro commits, as small commit as possible, so they are easy to trace the bug
* Commit only related files, no unnecesaary files
* Requires approval before adding new deps, try to use backward compatible polyfills

**(Inspired by Oracle Java Code Conventions, Microsoft C# Guide):**

* Prefer **explicit over implicit** behavior
* Follow language idioms and style conventions strictly
* Refactor when code smells (duplication, long methods, large classes, etc.) appear
* Fail fast: Throw exceptions when invariants break; don't silently swallow errors
* Do not optimize prematurely - measure first, optimize second

---

## Code Review Expectations

**(Amazon + Microsoft inspired):**

* Reviews should focus on **correctness, clarity, simplicity, testability, and maintainability**
* Don’t block PRs on personal style if it’s consistent with the codebase
* Leave comments that explain the *why*, not just the *what*
* Never approve code you don’t understand
* Be kind, constructive, and professional - code reviews are opportunities to mentor

---

## Testing Guidelines

**(Amazon: "You own what you ship")**

* All code must include **unit tests**, and ideally **integration and end-to-end tests**
* Tests must cover edge cases, not just the happy path
* Ensure test readability - test names should describe *intent*
* All tests must pass before merging
* Use mocks and stubs to isolate units, but don’t overuse mocking for internal logic

---

## Documentation Standards

* All exported/public methods should have doc comments
* Use meaningful commit messages: `[fix]: `, `[feat]: `, `[refactor]: `, `[docs]: `, `[refactor,fix]: `, `[test]: ` etc.
* README and setup instructions must be clear and beginner-friendly
* If the "why" behind a design isn’t obvious, document it
* Use architecture decision records (ADRs) for major decisions

---

## Security & Compliance

**(Inspired by Microsoft SDLC & Amazon Security Best Practices):**

* Validate all external inputs - never trust the client
* Use secure defaults: least privilege, HTTPS, strong authentication
* Always sanitize user-generated content
* Don’t log sensitive information (passwords, access tokens, etc.)
* Keep dependencies updated and minimal
* Run static/dynamic analysis tools regularly (e.g., ESLint, SonarQube, CodeQL)

---

## Development

Never try to run the server, I am running them in seperate terminals.