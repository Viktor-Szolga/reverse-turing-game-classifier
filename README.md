# Bachelor Thesis – Reverse Turing Game Classifier
This repository contains the code, trained models, and experiment configurations used for my bachelor thesis. It provides all scripts and data required to recreate the experimental results and plots described in the thesis.

## 📦 Installation
The project uses [Poetry](https://python-poetry.org/) for dependency management.

1. Clone this repository:
   ```bash
   git clone https://github.com/Viktor-Szolga/reverse-turing-game-classifier.git
   cd reverse-turing-game-classifier
   ```

2. Create the environment with Poetry:
   ```bash
   poetry install
   ```

   Alternatively, you can install from the frozen requirements file:
   ```bash
   pip install -r requirements_freeze.txt
   ```

## Running Experiments
The main entry point for all experiments is:
```bash
python main.py
```
This script:
- Trains all classifiers with different configurations as described in the thesis.
- Stores results and model checkpoints in the appropriate folders.

All experiment configurations are stored as YAML files in the `experiments/` folder.

## Trained Models
- All trained model state dictionaries are saved in the `trained_models/` directory.
- The final model selected in the thesis is:

  `trained_models/run20.pth`

## Data and Preprocessing
- **Message Encodings:** Precomputed encodings are included in the repository. (They were originally generated via `encode_data.ipynb`, which connects to a local server and therefore cannot be rerun directly.)
- **Game IDs:** To retrieve the game IDs of all games that only include players from the validation set when using the user split, run:
  ```bash
  python retrieve_game_ids.py
  ```

## Recreating Plots
Use `create_plots.py` to generate the visualizations from the thesis:

Make sure to select the correct runs according to the configurations in `experiments/`.

## Analysis and Evaluation
The repository also includes notebooks with the analysis and evaluation performed during the thesis:

- `analysis/data_analysis.ipynb`: Contains the data exploration and analysis steps.
- `analysis/classifier_evaluation.ipynb`: Contains the evaluation of the trained classifiers.

The results of these notebooks are already included in the repository for transparency and reproducibility.
