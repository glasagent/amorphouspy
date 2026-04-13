# Good First Issues

Good first issues are intentionally scoped tasks that a new contributor can complete without deep familiarity with the codebase. This page explains how to write one well.

## When to open a good first issue

Open a good first issue when you have a concrete, self-contained task that:

- Does not require understanding multiple subsystems at once.
- Has a clear, testable outcome.
- Leaves room for a contributor to make real decisions (not just copy-paste).

Typical candidates: adding a missing test, exposing a hard-coded value as a parameter, writing a how-to guide section, or implementing a small utility function.

## Using the template

When you open a new issue on GitHub, select **Good First Issue** from the template picker. The form has the following fields.

### Summary

One or two sentences. State what needs to be done, not why.

> *Add a `normalize` parameter to `compute_rdf()` that divides the histogram by the ideal-gas baseline.*

### Motivation

Explain the problem this solves. A contributor who understands the why will make better decisions when they hit ambiguity.

> *Users currently post-process the raw count histogram manually. Exposing normalization in the function removes boilerplate from every downstream notebook.*

### Relevant files / functions

Point to the **exact** file and function a contributor should read first. One or two lines is enough.

> `amorphouspy/src/amorphouspy/analysis/rdf.py` → `compute_rdf()`  
> `amorphouspy/src/tests/test_rdf.py` → existing test fixtures

If the contributor has to grep the whole codebase to find the entry point, the issue is under-specified.

### Suggested steps

Break the work into 3–6 numbered steps, each completable in under an hour. Include the test command at the end.

```
1. Read compute_rdf() and understand how the histogram is built.
2. Add a normalize: bool = True parameter.
3. Apply normalization inside the function when the flag is set.
4. Add a test in test_rdf.py that checks the normalized output sums correctly.
5. Run `pixi run test-lib` to verify all tests pass.
```

Avoid vague instructions like "update the function" — name the parameter, the file, the test.

### Definition of done

List the exact commands a reviewer will run to verify the issue is resolved.

```
- pixi run test-lib passes
- pixi run lint passes
- New or updated test covers the added behaviour
```

### Skills needed

Select all skills a contributor will actually need. Be honest — over-claiming "Python (basic)" for a task that touches LAMMPS input parsing will lead to frustrated contributors.

### Backward compatibility

State explicitly whether the change breaks existing code. If it does, the PR must carry the `breaking` label and the version bump must be `major`. If the change is additive (new parameter with a safe default), mark it as fully backward compatible.

### Contributor checklist

The checklist includes a **required** acknowledgment that tests will be added or updated. Do not open a good first issue for a task where testing is genuinely impossible — it sends the wrong signal to new contributors about project standards.

### Estimated effort

Be conservative. A "Small (< 2 hours)" estimate that takes a day will erode trust. If you are unsure, use "Medium".

## After the issue is opened

- Add the `good first issue` label (applied automatically by the template).
- If the change is additive, also add `enhancement` or `feature` so it appears in release notes.
- Comment with any additional context that did not fit the form — links to related issues, known pitfalls, or a short explanation of the data structures involved.
- When a contributor opens a PR, link it to the issue with `Closes #<number>` in the PR description.
