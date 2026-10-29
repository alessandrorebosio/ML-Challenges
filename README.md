# Machine Learning

Machine Learning exercises, organized in challenges that share a single dataset folder:

- `01-kNN&SVM`: multiclass classification on PenDigits with a PCA-SVC pipeline;
- `02-PetClassification`: cat vs dog image classification with HOG features and a multi-resolution SVC ensemble.

## Structure

```
ML-Challenges/
├── DBs/                      # datasets shared by all challenges
│   ├── PenDigits/            # 01-kNN&SVM
│   └── CaniGatti_ML18/       # 02-PetClassification
├── NN-Name/                  # one folder per challenge
│   ├── README.md             # task description and constraints
│   ├── src/
│   │   └── main.ipynb        # full solution, helpers included
│   └── *.txt                 # generated predictions (ignored by git)
├── requirements.txt
└── LICENSE
```

Each challenge is a single self-contained notebook in `src/`: it reads the data from `../../DBs/` and writes the predictions to the challenge folder, outside `src/`.

## Requirements

- macOS or Linux;
- Python 3.9, or a version compatible with the dependencies pinned in `requirements.txt`;
- `pip`.

## Creating and activating the `.venv`

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `deactivate` to leave the virtual environment; to activate it again, run `source .venv/bin/activate` from the root.

## VS Code setup

1. Open the repository folder in VS Code.
2. Run `Python: Select Interpreter` from the Command Palette (`Cmd+Shift+P`).
3. Choose `./.venv/bin/python`.
4. For the notebooks, select the same interpreter as the kernel in the top-right corner.

If the kernel does not show up, run with the environment active:

```bash
python -m ipykernel install --user \
	--name machine-learning-venv \
	--display-name "Python (Machine Learning .venv)"
```

## Running the challenges

Open `NN-Name/src/main.ipynb` in VS Code with the `.venv` kernel and run all cells in order. To use Jupyter in the browser instead, install it in the environment (`python -m pip install notebook`) and run `jupyter notebook` from the root.

Paths are relative to the notebook's `src/` folder, which is the default working directory in both VS Code and Jupyter. The outputs are written to the challenge folder:

| Challenge | Output |
| --- | --- |
| `01-kNN&SVM` | `Es3Predictions.txt` |
| `02-PetClassification` | `Es4Predictions.txt` |

All notebooks share the same structure: `Setup` (imports and configuration), `Helper Functions`, `Data Loading`, `Model`, `Model Evaluation` (cross-validation) and `Prediction and Export`.

In `02-PetClassification` the first run is slower, because the loaded and resized images are cached in `02-PetClassification/Experiments/`.

## Ignored files

- predictions (`*.txt` outside `DBs/`);
- joblib caches (`Experiments/`);
- the virtual environment (`.venv/`) and system/editor files.

## Quick environment check

```bash
python -c "import numpy, pandas, sklearn, skimage, matplotlib, joblib; print('Environment ready')"
```
