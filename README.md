# Homework 7: LLM, TensorFlow, JAX

| File | Question |
|---|---|
| `model.py` | the GPT model from Lecture 15, used by Q1 |
| `lm_utils.py` | the data, `TextDataset` and optimizer/LR schedule from the Lecture 15/16 notebook, used by Q1 |
| `data/shakespeare.txt` | the Lecture 15 text dataset, used by Q1 |
| `Q2.ipynb` | Q2: a `tf.keras` model for Fashion-MNIST, and TensorBoard |
| `Q3.ipynb` | Q3: gradient-based optimization through a bouncing-ball simulator in JAX |
| `agent-solutions/` | the coding agent's solution for Q2-3 |
| `.agents/skills/tensorflow/` | the `/tensorflow` skill for the coding agent in Q2-3 |

For Q1, write your own training script in this folder, using `model.py` and
`lm_utils.py`. See `lm_utils.py` for how the pieces fit together.

Fashion-MNIST (Q2) is downloaded by Keras into `~/.keras/datasets/`.

