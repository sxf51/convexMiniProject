# Mini-Project: MPC-Based Path Planning and Following for Autonomous Vehicles

[![CI](https://github.com/sxf51/convexMiniProject/actions/workflows/ci.yml/badge.svg)](https://github.com/sxf51/convexMiniProject/actions/workflows/ci.yml)

This repository contains the mini-project for **ELEN90026 Introduction to Optimisation**
(Semester 2, 2026). The goal is to design Model Predictive Control (MPC) based path
planning and path following controllers for an autonomous vehicle, and to solve the
resulting quadratic programs (QPs) with optimisation algorithms taught in the course.

## Overview

The vehicle is modelled as a **double integrator** on a 2D track:

```text
p_dot = v,   v_dot = a
```

The continuous system is discretised with Euler's approximation, and process noise is
added for simulation. At each time step, an MPC problem minimises tracking error and
control effort over a finite prediction horizon:

```text
min  sum_i ||x(i) - x_bar(i)||_Q^2 + ||u(i)||_R^2 + ||x(N) - x_bar(N)||_Q^2
s.t. x(0) = x_hat
     x(i+1) = A x(i) + B u(i)
     u_min <= u(i) <= u_max
     d_min(i) <= c(i)^T x(i) <= d_max(i)
```

The assignment is split into two parts:

- **Part 1 - Path following** (Tasks 1-3): solve the sequential MPC problem with no
  constraints, input constraints, and both input and state constraints.
- **Part 2 - Planning and following** (Tasks 4-5): add collision-avoidance constraints
  for path planning, then track the planned path.

The full specification is available in
[docs/Project_part_1-1.pdf](docs/Project_part_1-1.pdf).

## Repository Structure

```text
.
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── docs/
│   └── Project_part_1-1.pdf   # Assignment specification
├── tests/
│   └── test_main.py           # Smoke tests
├── main.py                    # Entry point
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependency versions
└── README.md
```

## Getting Started

The project uses [uv](https://github.com/astral-sh/uv) as its Python package and
environment manager. Python 3.13 or newer is required.

```bash
uv sync
```

Run the entry point:

```bash
uv run python main.py
```

## Dependencies

Runtime dependencies are declared in `pyproject.toml`:

- [CasADi](https://web.casadi.org/) - algorithmic differentiation and nonlinear
  optimisation.
- [Gurobi](https://www.gurobi.com/) - commercial quadratic programming solver.

> **Note on Gurobi:** `gurobipy` installs without a license, but solving models requires a
> valid Gurobi or Web License Service (WLS) license. The CI pipeline runs only smoke tests
> and does not invoke Gurobi.

## Development

The development dependency group provides [pytest](https://pytest.org/) and
[Ruff](https://docs.astral.sh/ruff/).

Run the test suite:

```bash
uv run pytest
```

Lint and format the code:

```bash
uv run ruff check .
uv run ruff format .
```

## CI/CD

GitHub Actions runs on every push and pull request to `main`/`master`. The
[workflow](.github/workflows/ci.yml) installs dependencies with `uv sync --all-groups`,
then runs:

1. `ruff check .`
2. `ruff format --check .`
3. `uv run pytest`

## Contributors

The following individuals have contributed to this project:

[![Contributors](https://contrib.rocks/image?repo=sxf51/convexMiniProject&max=1000)](https://github.com/sxf51/convexMiniProject/graphs/contributors)
