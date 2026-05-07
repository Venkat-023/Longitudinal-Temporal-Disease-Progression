# Train With VS Code + Colab GPU

I could not start a Colab GPU run from this local Codex session because VS Code
does not currently expose a Colab-authenticated runtime here. The repo is ready
for the VS Code Colab/Jupyter workflow, though.

## What Is Installed Locally

VS Code CLI is available and these relevant extensions are installed:

- `ms-python.python`
- `ms-toolsai.jupyter`
- `ms-toolsai.jupyter-renderers`

I did not find a Colab-specific extension in `code --list-extensions`.

## Run From VS Code

1. Install/open your preferred Colab extension in VS Code.
2. Open:

```text
notebooks/mimiciv_colab_gpu_training.ipynb
```

3. Select a Google Colab GPU runtime.
4. Run the cells from top to bottom.

The full training cell runs:

```bash
python src/run_full_colab_training.py --zip data/mimic_iv_raw/mimic-iv-2-1.zip --chunk 100000 --seq_len 6 --epochs 80 --batch_size 128 --hidden_size 128 --patience 12
```

That command deletes generated outputs first, preprocesses the complete
MIMIC-IV ZIP, trains BiLSTM, BiGRU, and Transformer, then writes:

```text
data/preprocessed/
models/
visualizations/
results.json
```

## Important

Do not use the existing small local `data/preprocessed` arrays for final
training. The Colab pipeline intentionally rebuilds them from the full ZIP.
