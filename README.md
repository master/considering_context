# Considering Context: When World Models Need Context Encoders

Requires Python 3.11 and CUDA. Install dependencies with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra carl
```

Run a method on the default CARL Walker setting:

```sh
uv run --extra carl python train.py context@model.context=hidden
uv run --extra carl python train.py context@model.context=dali_s
uv run --extra carl python train.py context@model.context=carried_state
uv run --extra carl python train.py context@model.context=lilac
uv run --extra carl python train.py context@model.context=crssm
```

Based on the [R2-Dreamer](https://github.com/NM512/r2dreamer) codebase. Released under the MIT license.
