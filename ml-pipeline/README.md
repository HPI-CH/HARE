# Note-Paper Code Submission

## Setup
Create a virtualenv with the required packages with the provided `environment.yml`

Run the following command in a terminal in this folder to create one.

```
conda env create --file=environment.yml
```

## Running the code

Make sure that you have enough RAM available (in the best case more than 32GB). Then switch to the created environment (default name: `note-paper` -> `conda activate note-paper`) and run `python src/main.py` to start the script.

Afterwards, you should have a file `results.csv` in this folder with all model information and results.