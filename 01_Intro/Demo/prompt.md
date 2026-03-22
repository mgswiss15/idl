# Prompting Claude (Sonet 4.6)

## Me

> I want to classify CIFAR-10 images. Give me the PyTorch code to train and test the classification models. This should be in multiple scripts for loading, training, evaluating and for the models themselves - simple MLP and CNN. The models shall be compared in terms for their performance - I need a code for this as well. I will also need instructions in a simple readme.md file how to train and evaluate the models. And this shall all be reproducible if I need to do it again and get the same results.

## Claude

Now let me create all the scripts systematically:Here's the complete package — 7 Python files + README. Here's a quick overview of what you get:

**File structure:**
- `data_loader.py` — downloads CIFAR-10, applies the right transforms per model type (CNN keeps spatial structure, MLP flattens), and returns reproducible train/val/test splits
- `models.py` — `MLP` (3072 → 1024 → 512 → 256 → 10) and `CNN` (VGG-style, 3 conv stages with 64/128/256 channels), both with BatchNorm and Dropout; plus a `get_model()` factory
- `utils.py` — `set_seed()` that pins Python, NumPy, PyTorch, and cuDNN to guarantee reproducibility
- `train.py` — AdamW + cosine annealing scheduler, label smoothing, saves best checkpoint by val accuracy, logs history to JSON
- `evaluate.py` — loads a checkpoint, runs test-set inference, prints per-class classification report, saves normalised confusion matrix PNG
- `compare.py` — loads both checkpoints, plots training curves and per-class accuracy bars side-by-side, saves a text summary table
- `README.md` — setup, quickstart, all CLI flags documented, expected results, architecture details

**To run, just follow this order:**
```bash
pip install -r requirements.txt
python train.py --model cnn
python train.py --model mlp
python evaluate.py --model cnn
python evaluate.py --model mlp
python compare.py
```

You can expect roughly ~55–58% for the MLP and ~87–90% for the CNN.