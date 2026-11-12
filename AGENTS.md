# Rust Import and Module Declaration Order

Organize the top of each file in this order:

1. Standard library imports (`std`, `core`, `alloc`).
2. Third-party imports, including third-party `pub use` statements.
3. Regular local module declarations (`mod name;`).
4. Local imports and re-exports (`use self::…`, `use super::…`, `use crate::…`, `pub use self::…`).

Separate these groups with a blank line. Use explicit `self::` paths when re-exporting items from local child modules.

Within each import group, place `pub use` statements before ordinary `use` statements, with a blank line between them. Keep the standard library, third-party, and local group order above.

Place all test modules at the end of the file, after production items. This includes both external declarations (`#[cfg(test)] mod tests;`) and inline test modules. Do not group them with regular module declarations or imports.

Keep all test-only imports, helper functions, types, constants, statics, and
`impl` blocks inside test modules. Do not place individual `#[cfg(test)]`
imports or helper items in production scope. Put test-only methods in a
separate `impl` block inside the test module, rather than in the production
`impl`. Shared test helpers belong in the crate root's test module and should
use only the visibility needed by their test callers.

```rust
use std::io;

use external_crate::ExternalType;

mod component;

pub use self::component::Component;

use crate::Config;

#[cfg(test)]
mod tests;
```

# Rust Module File Layout

- If a module contains only its own implementation and tests, keep both in a
  single `module_name.rs` file, with an inline test module. Split out the tests
  only when that file becomes exceptionally long.
- Keep the crate root's tests and shared test helpers in an inline test module
  in `lib.rs` as well. Do not create a separate `test.rs` or `tests.rs` unless
  the test code becomes exceptionally long.
- If a module has any non-test child modules, use a `module_name/` directory
  with a `mod.rs` file and files for its child modules.

# Function and Test Organization

- Order functions from callers to their helpers. Place a dedicated helper
  immediately after the function that uses it; place a shared helper after
  its related callers.
- Apply this order within implementation blocks and test modules as well.
  Keep test modules at the end of the file and preserve existing item scopes.
- Unit tests should focus on the functions and types implemented in their
  own file. Keep iterator behavior tests with the iterator implementation
  and vtable dispatch tests with the vtable implementation.
- Make each test's intent visible in its name, inputs, and assertions. Test one
  behavior at a time, and use explicit expected results for concrete scenarios.
- Keep the scenario and substantive assertions in the test itself. Limit shared
  helpers to clearly defined setup, such as constructing a needle or allocating
  guarded memory; do not hide unrelated scenarios behind a general callback.

# Static and Constant Scope

- Declare `static` and `const` items in the function or block that uses them
  when they do not need a wider scope.
- Keep shared items in the smallest scope containing all their uses. Use
  module scope only when the item needs to be shared across functions.
- Preserve static storage and sharing semantics when moving declarations.

# Documentation Requirements

- Apply these requirements throughout the project, not just to particular types
  or modules.
- Write API documentation for callers: explain what the operation does, what it
  returns, and how it changes observable state.
- Do not repeat guarantees inherent in the type's normal operation in every
  method's documentation. Focus on the operation's purpose, meaningful choices,
  and exceptions or failure effects callers need to know.
- Use the documentation of analogous standard library or established library
  APIs as a model. Describe this project's actual behavior; do not copy
  guarantees that do not apply.
- Use plain, concrete language. Explain parameter meanings, return values, and
  observable effects precisely. Prefer short before-and-after examples over
  abstract descriptions.
- Do not expose implementation details in public documentation unless they are
  necessary to use the API correctly. Keep implementation-specific reasoning
  in ordinary code comments. The README may explain the Shufti algorithm and
  table construction with worked examples.
- Describe behavior directly instead of translating assignments or branch
  conditions into prose. Express relevant conditions in plain language and
  preserve meaningful guarantees and failure effects. Keep exact bounds or
  formulas when they help callers use the API correctly or safely.
- Document observable guarantees and necessary constraints, including relevant
  panic conditions, errors, ownership, borrowing, and safety requirements. Do
  not remove required `# Safety` documentation or omit facts callers need to use
  an API safely.
- Keep internal safety reasoning in `// SAFETY:` comments. Explain why the
  operation is valid, rather than merely asserting that it is safe.
- When changing documentation, preserve code behavior and verify runnable
  examples with documentation tests. Keep examples short and focused on the
  operation being explained.
- When renaming an API, update related rustdoc, internal comments, README
  examples, and test names together. Use consistent terminology for the same
  operation throughout the project.

# Commit Messages

- Prefix commit subjects with a change type, using `type: description` or
  `type(scope): description`.
- Choose the type that describes the change: `feat`, `fix`, `docs`, `test`,
  `refactor`, `perf`, `ci`, `build`, or `chore`.
- Write concise English descriptions of the actual change. For example,
  `feat: reconstruct matching byte sets` or `ci: test stable and MSRV`.
