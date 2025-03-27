# Experiments Summary

This document summarizes a series of experiments conducted on Hindi language models using the SentencePiece BPE algorithm. The experiments include details about configurations, training times, and observations for future reference. My baseline is to get a BLEU score of 30 with a smaller sized model on a test set which I do not touch until I feel like the model is ready. The SOTA seems to be ~40 in 2022 for BLEU.

**General Settings:**

- **BPE Algorithm:** SentencePiece
- **Vocabulary Size:** 8000

---

## Experiment: hindi_drop_warm_smoothen

**Purpose:**  
Establish a baseline by training a full-size model for 10 epochs.

**Config Settings:**

- **Model Architecture:**
  - `D_MODEL`: 512
  - `N_HEADS`: 8
  - `N_ENCODERS`: 6
  - `N_DECODERS`: 6
  - `FF_HIDDEN`: 2048
- **Training Parameters:**
  - `DROPOUT`: 0.1
  - `SEQ_LEN`: 140
  - `BATCH_SIZE`: 64
  - `TRAIN_RATIO`: 0.8
  - `NUM_EPOCHS`: 10
- **Other Configs:**
  - **Warmup:** 2000 steps
  - **ngpus:** 1

**Observations:**

- Total training time: ~20 minutes.

---

## Experiment: hindi_drop_warm_smoothen_small

**Purpose:**  
Reduce model size (to approximately 9M parameters) due to long training times with the larger model, and increase training samples to achieve better convergence.

**Config Settings:**

- **Model Architecture:**
  - `D_MODEL`: 256
  - `N_HEADS`: 4
  - `N_ENCODERS`: 2
  - `N_DECODERS`: 2
  - `FF_HIDDEN`: 2048
- **Training Parameters:**
  - `DROPOUT`: 0.1
  - `SEQ_LEN`: 140
  - `BATCH_SIZE`: 128
  - `TRAIN_RATIO`: 0.8
  - `NUM_EPOCHS`: 450
- **Other Configs:**
  - **Warmup:** 2000 steps
  - **ngpus:** 1
  - **Total Samples:** 125K

**Comments:**

- The model reached the 12-hour training limit on the HPC.
- Due to this, the implementation was modified to log test/train curves and intermediate outputs to a text file, ensuring progress is not lost.

---

## Experiment: hindi_drop_warm_smoothen_small_3

**Purpose:**  
Same as **hindi_drop_warm_smoothen_small**, but with intermediate outputs written to a file.

**Config Settings:**  
_Same as_ **hindi_drop_warm_smoothen_small**

**Other Configs:**  
_Same as_ **hindi_drop_warm_smoothen_small**

**Comments:**

- Intermediate outputs are logged to a file, as in the previous experiment.

---

## Experiment: multi_hindi_drop_warm_smoothen_small

**Purpose:**  
Test multi-GPU training using `nn.DataParallel` to see if 4 GPUs provide the expected linear time speedup.

**Config Settings:**

- **Model Architecture:**
  - `D_MODEL`: 256
  - `N_HEADS`: 4
  - `N_ENCODERS`: 2 
  - `N_DECODERS`: 2
  - `FF_HIDDEN`: 2048
- **Training Parameters:**
  - `DROPOUT`: 0.1
  - `SEQ_LEN`: 140
  - `BATCH_SIZE`: 128
  - `TRAIN_RATIO`: 0.8
  - `NUM_EPOCHS`: 100
  - `LR`: 0.00005
- **Other Configs:**
  - **Warmup:** 500 steps
  - **ngpus:** 4
  - **Total Samples:** 125K

**Comments:**

- Despite using 4 GPUs, training still took around 2:02 hours.
- GPU utilization details:

  | Node | GPU ID | Requested | Used | Efficiency | Available Memory | Used Memory |
  |------|--------|-----------|------|------------|------------------|-------------|
  | k105 | 1      | 1         | 0.44 | 44%        | 32768 MiB        | 1.86G       |
  | k105 | 0      | 1         | 0.55 | 55%        | 32768 MiB        | 4.39G       |
  | k105 | 3      | 1         | 0.37 | 37%        | 32768 MiB        | 1.87G       |
  | k105 | 2      | 1         | 0.40 | 40%        | 32768 MiB        | 1.84G       |
  
  **Total:** 4 GPUs, overall ~44% utilization.

---

## Experiment: multi_hindi_drop_warm_smoothen_small_1

**Purpose:**  
Investigate if increasing the sequence length and training set size while maintaining overall training time can increase per-GPU utilisation.

**Config Settings:**

- **Model Architecture:**
  - `D_MODEL`: 256
  - `N_HEADS`: 4
  - `N_ENCODERS`: 2
  - `N_DECODERS`: 2
  - `FF_HIDDEN`: 2048
- **Training Parameters:**
  - `DROPOUT`: 0.1
  - `SEQ_LEN`: 140
  - `BATCH_SIZE`: 256
  - `TRAIN_RATIO`: 0.8
  - `NUM_EPOCHS`: 100
  - `LR`: 0.00005
- **Other Configs:**
  - **Warmup:** 2000 steps
  - **ngpus:** 4
  - **Total Samples:** 250K

**Comments:**

- This experiment aims to improve per-GPU utilisation by using a larger batch size and more training samples without extending the total training time. It does not seem like the utilisation was affected much per gpu. The total runtime was 3:09 hrs. 
--------------------------------------------------------------------------------
|              |             GPUs              |            Memory             |
--------------------------------------------------------------------------------
| Node  GPU ID | Requested   Used   Efficiency | Available   Used              |
| k105    1    |     1       0.53       53%    | 32768 MiB   2.51G             |
| k105    0    |     1       0.67       67%    | 32768 MiB   7.72G             |
| k105    3    |     1       0.44       44%    | 32768 MiB   2.46G             |
| k105    2    |     1       0.49       49%    | 32768 MiB   2.5G              |
--------------------------------------------------------------------------------
|Total         |     4       2.13     53.25%   |                               |
--------------------------------------------------------------------------------



### multi_hindi_drop_warm_smoothen_small_2

(this is with _3 in the .txt file but the Model is _2.)
D_MODEL: 256
N_HEADS: 4
N_ENCODERS: 2 # 6
N_DECODERS: 2 # 6
FF_HIDDEN: 2048
DROPOUT: 0.1
SEQ_LEN: 140
BATCH_SIZE: 1024
TRAIN_RATIO: 0.8
NUM_EPOCHS: 400
LR: 0.00005
DROPOUT: 0.1
- **Other Configs:**
  - **Warmup:** 2000 steps
  - **ngpus:** 8
  - **Total Samples:** 550K

## multi_hindi_small
Testing to see if the optimiser works with the new state dicts. I was not updating the Adam optimiser momentum params when I was re-initialising to re-train the model. 

D_MODEL: 256
N_HEADS: 4
N_ENCODERS: 2 # 6
N_DECODERS: 2 # 6
FF_HIDDEN: 2048
DROPOUT: 0.1
SEQ_LEN: 140
BATCH_SIZE: 1024
TRAIN_RATIO: 0.8
NUM_EPOCHS: 400
LR: 0.00005
DROPOUT: 0.1
- **Other Configs:**
  - **Warmup:** 2000 steps
  - **ngpus:** 4
  - **Total Samples:** 550K

## multi_hindi_metrics
Maybe batch loss wasn't correctly being calculated, I was doing the mean and then averaging the mean of per batch, I've changed it to sum and and average per token.

D_MODEL: 256
N_HEADS: 4
N_ENCODERS: 2 # 6
N_DECODERS: 2 # 6
FF_HIDDEN: 2048
DROPOUT: 0.1
SEQ_LEN: 140
BATCH_SIZE: 1024
TRAIN_RATIO: 0.8
NUM_EPOCHS: 400
LR: 0.00005
DROPOUT: 0.1
- **Other Configs:**
  - **Warmup:** 2000 steps
  - **ngpus:** 4
  - **Total Samples:** 550K

## multi_hindi_restart
These sets of experiments aim to solve a very big problem I'm encountering. When I save a model and all the optimiser dict values, even despite that the training and testing loss do not start from the same point where it stopped. 

D_MODEL: 256
N_HEADS: 4
N_ENCODERS: 2 # 6
N_DECODERS: 2 # 6
FF_HIDDEN: 2048
DROPOUT: 0.1
SEQ_LEN: 140
BATCH_SIZE: 1024
TRAIN_RATIO: 0.8
NUM_EPOCHS: 400
LR: 0.00005
DROPOUT: 0.1
- **Other Configs:**
  - **Warmup:** 2000 steps
  - **ngpus:** 4 (H200s)
  - **Total Samples:** 550K
