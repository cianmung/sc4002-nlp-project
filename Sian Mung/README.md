# M1–M3 Joint Model Ensemble

## Setup

1. Install Python 3 and Jupyter Notebook.
2. Install the required packages:

   ```bash
   pip install torch numpy pandas tokenizers notebook ipython
   ```

3. Put `final_M1_M2_M3_joint_ensemble.ipynb` and `en-US.jsonl` in the same folder.
4. If training from scratch, download GloVe from [Stanford NLP](https://nlp.stanford.edu/data/glove.6B.zip). Extract `glove.6B.zip` and put `glove.6B.300d.txt` in the same folder as the notebook. You do not need the GloVe file when loading all three saved models.
5. Open the notebook with `jupyter notebook`.

## Train from scratch

1. In the configuration cell, set:

   ```python
   MODE = {"M1": "train", "M2": "train", "M3": "train"}
   M1_INIT_MODE = "train"
   RUN_TEST_ONCE = False
   ```

2. Run all cells from top to bottom.
3. The notebook trains M1, M2 and M3, then evaluates their ensemble on the development set.
4. Training times and validation results are printed and saved to the output folder.

## Load saved models

1. Put these files in `final_M1_M2_M3_outputs/`:
   - `M1_best.pt`
   - `M2_best.pt`
   - `M3_best.pt`
   - `M3_bpe_tokenizer.json`
2. In the configuration cell, set:

   ```python
   MODE = {"M1": "load", "M2": "load", "M3": "load"}

   OUTPUT_DIR = Path("final_M1_M2_M3_outputs")
   M1_INIT_MODE = "load"

   LOAD_PATH = {
       "M1_init": OUTPUT_DIR / "M1_init_seed42_best.pt",
       "M1": OUTPUT_DIR / "M1_best.pt",
       "M2": OUTPUT_DIR / "M2_best.pt",
       "M3": OUTPUT_DIR / "M3_best.pt",
   }

   BPE_LOAD_PATH = OUTPUT_DIR / "M3_bpe_tokenizer.json"
   RUN_TEST_ONCE = False
   ```

3. Run all cells from top to bottom. The notebook loads the saved models without retraining and evaluates the ensemble.

## Final evaluation

After selecting the final models, set `RUN_TEST_ONCE = True` and rerun the notebook to evaluate the held-out test set. Do not change the models based on test results.

## Output files

Results are saved in `final_M1_M2_M3_outputs/`, including model checkpoints, the BPE tokenizer, validation and test results (when enabled), training times, and `M1_M2_M3_ensemble_bundle.pt`.

The notebook reloads the saved ensemble bundle and checks that its development results match before running the optional test evaluation.
