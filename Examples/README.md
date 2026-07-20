# Running the examples with UV

From the repository root, install the locked project environment once:

```bash
uv sync
```

Run an example through that environment:

```bash
uv run python Examples/example_new_architecture.py
```

Replace the filename with any other script in this directory, for example:

```bash
uv run python Examples/example_parsim_demo.py
uv run python Examples/example_gen.py
uv run python Examples/Ex_SS.py
```

Some examples create Matplotlib figures and keep a window open until it is
closed. Run commands from the repository root so local package imports resolve
consistently.
