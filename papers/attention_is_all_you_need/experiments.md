# Experiments Summary

This document summarizes a series of experiments conducted on Hindi language models using the SentencePiece BPE algorithm. The experiments include details about configurations, training times, and observations for future reference. My baseline is to get a BLEU score of 30 with a smaller sized model on a test set which I do not touch until I feel like the model is ready. The SOTA seems to be ~40 in 2022 for BLEU.

**General Settings:**

- **BPE Algorithm:** SentencePiece
- **Vocabulary Size:** 8000
- **Training Set Size:** 550K

---

**Learnings:**

These are just some basic mistakes I made while coding the first transformer.
- Logging test/train curves and intermediate outputs to a text file, every 10 epochs so that when I hit 12 hour HPC limit, my progress was not lost.
- Using multi-GPU training using `nn.DataParallel`. It did not provide a linear time speed up as I would expect and this could be due to the overall latency of copy the weights of a 11M param model. There was roughly ~50 utilisation of the 4 GPUs and about a 50 % speedup on training.
- When attempting to restart model training from the saved checkpoint, I was not updating the updating the Adam optimiser momentum params. Also changed loss calculation on a token-level, not batch level. 
- Did not set all random seeds so when I was using a model checkpoint to start re-training, there was training and validation set mixing which was causing the validation to sharply fall and then rise with more training and the training loss to sharply rise after a restart and then fall!
- Also set a minimum LR, otherwise with the sqrt decay of LR used in the paper, the model's learning rate after 300 epochs would be way too small.
- Ran a 1000 epoch training and the performance increase stagnated around ~300 epochs. With the given batch size and training set, this was rougly 130k steps (for 300 epochs). As such, there was no point training a model for longer than 300 epochs. This model achieved a BLEU score of 45 on a testing set of 100K samples at both 300 epochs and 1000 epochs. As such, there was no use training a model for any longer. 
---
The convention for model naming is:

`model_{N_HEADS}_{D_MODEL}_{FF_HIDDEN}_{N_ENCODERS}_{N_DECODERS}`


## Number of Heads Experiments
All models were ran on the same GPU for 300 epochs of training. My first set of experiments was seeing what difference changing the n_heads made on training.  I kept the total parameter count the same to make sure all the differences I could observe was due to changing the number of heads and not due to increased/decreased model capacity. As in the original paper, the dimensionality of each head was adjusted based on D_MODEL // N_HEADS so that made the parameter count exactly the same.


### N_Heads = 4
- D_MODEL: 256
- N_HEADS: 4
- N_ENCODERS: 2 # 6
- N_DECODERS: 2 # 6
- FF_HIDDEN: 2048
- DROPOUT: 0.1
- SEQ_LEN: 140
- BATCH_SIZE: 1024
- TRAIN_RATIO: 0.8
- NUM_EPOCHS: 300
- DROPOUT: 0.1
- **Other Configs:**
  - **Warmup:** 2000 steps
  - **ngpus:** 4 (H200s)

