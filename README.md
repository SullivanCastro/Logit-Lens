# Imports

## Installation of libraries


```python
%pip install transformers -q
%pip install tqdm -q
```

## Importation of usual libraries



```python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
```

# Set up the model

## Importation of the GPT2 tokenizer and model 


```python
""" I import a model gpt2 pretrained model. The model is on the training mode, 
so gpt2 tries to predict each word from the precedent words."""
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

from transformers import AutoModelForSequenceClassification
model = GPT2Model.from_pretrained("gpt2")
```

## Implementation of the layer scrapping


```python
NB_LAYERS = 11 

output_saving = []
index = []

"""We define the hook to retrieve the different output of each block. The hook 
will be automatically appeal and each output will be append to the 
output_saving list. To garantee the possibility to decode the output we have 
to apply the last layer norm."""
for k in range(NB_LAYERS):

  def forward_hook(self, input, output):
    # Tensor of size (1, SEQ_LEN, TOKEN_ID)
    output_saving.append(model.ln_f(output[0]))


  model.h[k].register_forward_hook(forward_hook)
  index.append(f'gpt block {k}')
```

### BONUS : illustration of the model architecture


```python
"""architecture of the model"""
list(model.modules())
```




    [GPT2Model(
       (wte): Embedding(50257, 768)
       (wpe): Embedding(1024, 768)
       (drop): Dropout(p=0.1, inplace=False)
       (h): ModuleList(
         (0): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (1): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (2): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (3): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (4): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (5): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (6): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (7): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (8): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (9): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (10): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
         (11): GPT2Block(
           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (attn): GPT2Attention(
             (c_attn): Conv1D()
             (c_proj): Conv1D()
             (attn_dropout): Dropout(p=0.1, inplace=False)
             (resid_dropout): Dropout(p=0.1, inplace=False)
           )
           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
           (mlp): GPT2MLP(
             (c_fc): Conv1D()
             (c_proj): Conv1D()
             (act): NewGELUActivation()
             (dropout): Dropout(p=0.1, inplace=False)
           )
         )
       )
       (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
     ),
     Embedding(50257, 768),
     Embedding(1024, 768),
     Dropout(p=0.1, inplace=False),
     ModuleList(
       (0): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (1): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (2): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (3): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (4): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (5): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (6): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (7): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (8): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (9): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (10): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
       (11): GPT2Block(
         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (attn): GPT2Attention(
           (c_attn): Conv1D()
           (c_proj): Conv1D()
           (attn_dropout): Dropout(p=0.1, inplace=False)
           (resid_dropout): Dropout(p=0.1, inplace=False)
         )
         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
         (mlp): GPT2MLP(
           (c_fc): Conv1D()
           (c_proj): Conv1D()
           (act): NewGELUActivation()
           (dropout): Dropout(p=0.1, inplace=False)
         )
       )
     ),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     GPT2Block(
       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (attn): GPT2Attention(
         (c_attn): Conv1D()
         (c_proj): Conv1D()
         (attn_dropout): Dropout(p=0.1, inplace=False)
         (resid_dropout): Dropout(p=0.1, inplace=False)
       )
       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
       (mlp): GPT2MLP(
         (c_fc): Conv1D()
         (c_proj): Conv1D()
         (act): NewGELUActivation()
         (dropout): Dropout(p=0.1, inplace=False)
       )
     ),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2Attention(
       (c_attn): Conv1D()
       (c_proj): Conv1D()
       (attn_dropout): Dropout(p=0.1, inplace=False)
       (resid_dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     Dropout(p=0.1, inplace=False),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True),
     GPT2MLP(
       (c_fc): Conv1D()
       (c_proj): Conv1D()
       (act): NewGELUActivation()
       (dropout): Dropout(p=0.1, inplace=False)
     ),
     Conv1D(),
     Conv1D(),
     NewGELUActivation(),
     Dropout(p=0.1, inplace=False),
     LayerNorm((768,), eps=1e-05, elementwise_affine=True)]



# Implementation of the algorithm

## Run the test


```python
input = "What is the capital of the US"

"""we tokenize the word"""
encoded_input = tokenizer.encode(input, return_tensors='pt')

"""I don't need to keep the result of the last layer because this layer 
puts often ',' as a precaution"""
model(*encoded_input) 
"""the hook is activated during the evaluation of the model"""
```




    'the hook is activated during the evaluation of the model'



## Decode algorithm implementation


```python
def decode(output):
  """This function decodes the tokens and projets them into the words space"""
  W = model.wte.weight
  W_t = torch.transpose(W, 0, 1)
  final, logits = [], []
  for i in range(output.shape[0]):
    aux = output[i]@W_t
    final.append(torch.argmax(aux))
    logits.append(torch.max(output[i]).item())

  for i in range(len(final)):
    final[i] = tokenizer.decode(final[i])
  return final, logits
```

## Shape the results


```python
"""sentence_shattered is the list of the different tokens of the input"""
sentence_shattered = tokenizer.encode(input)
for i in range(len(sentence_shattered)):
  sentence_shattered[i] = tokenizer.decode(sentence_shattered[i])

"""I stock the differents decoded words in words_df and their logits in 
logits_df"""
words_df, logits_df = [sentence_shattered], [[0]*len(sentence_shattered)]
for i, element in enumerate(tqdm(output_saving[:])):
  final, logits = decode(element[0])
  words_df.append(final)
  logits_df.append(logits)

"""I shape the data in a pandas dataframe """
index.insert(0, "sentence")
words_df = pd.DataFrame(words_df, index=index)
logits_df = pd.DataFrame(logits_df, index=index)
words_df = words_df.dropna(axis=1)
logits_df = logits_df.dropna(axis=1)
```

    100%|██████████| 11/11 [00:01<00:00,  7.60it/s]


# Data analysis

## Logits dataframe


```python
logits_df
```





  <div id="df-e05b51c4-1181-43fb-88a1-5c482a50dedf">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sentence</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>gpt block 0</th>
      <td>7.787462</td>
      <td>20.594513</td>
      <td>9.138568</td>
      <td>24.357695</td>
      <td>9.559299</td>
      <td>6.387348</td>
      <td>9.397331</td>
    </tr>
    <tr>
      <th>gpt block 1</th>
      <td>7.040924</td>
      <td>18.778166</td>
      <td>7.534872</td>
      <td>23.300247</td>
      <td>6.357446</td>
      <td>8.246294</td>
      <td>7.242647</td>
    </tr>
    <tr>
      <th>gpt block 2</th>
      <td>6.761999</td>
      <td>17.837879</td>
      <td>8.612438</td>
      <td>30.378376</td>
      <td>8.686580</td>
      <td>12.181510</td>
      <td>17.720240</td>
    </tr>
    <tr>
      <th>gpt block 3</th>
      <td>6.818307</td>
      <td>23.713839</td>
      <td>11.395821</td>
      <td>37.030491</td>
      <td>10.413383</td>
      <td>15.532322</td>
      <td>23.015656</td>
    </tr>
    <tr>
      <th>gpt block 4</th>
      <td>6.862739</td>
      <td>30.430954</td>
      <td>19.140562</td>
      <td>36.668694</td>
      <td>12.113029</td>
      <td>15.560987</td>
      <td>24.589489</td>
    </tr>
    <tr>
      <th>gpt block 5</th>
      <td>6.930362</td>
      <td>32.779774</td>
      <td>21.728615</td>
      <td>37.152084</td>
      <td>8.199759</td>
      <td>13.397869</td>
      <td>21.074200</td>
    </tr>
    <tr>
      <th>gpt block 6</th>
      <td>7.022958</td>
      <td>34.188282</td>
      <td>24.558947</td>
      <td>43.888027</td>
      <td>7.270880</td>
      <td>13.949028</td>
      <td>28.208420</td>
    </tr>
    <tr>
      <th>gpt block 7</th>
      <td>7.173609</td>
      <td>44.923492</td>
      <td>32.673634</td>
      <td>45.391575</td>
      <td>5.735794</td>
      <td>20.303862</td>
      <td>24.776602</td>
    </tr>
    <tr>
      <th>gpt block 8</th>
      <td>7.284205</td>
      <td>53.881325</td>
      <td>50.814537</td>
      <td>42.364552</td>
      <td>15.902967</td>
      <td>18.179041</td>
      <td>25.080923</td>
    </tr>
    <tr>
      <th>gpt block 9</th>
      <td>7.672854</td>
      <td>65.964958</td>
      <td>66.906639</td>
      <td>40.341671</td>
      <td>28.846561</td>
      <td>21.481279</td>
      <td>32.710480</td>
    </tr>
    <tr>
      <th>gpt block 10</th>
      <td>9.615498</td>
      <td>135.350769</td>
      <td>141.646652</td>
      <td>103.498466</td>
      <td>94.174065</td>
      <td>104.911774</td>
      <td>96.959976</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e05b51c4-1181-43fb-88a1-5c482a50dedf')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e05b51c4-1181-43fb-88a1-5c482a50dedf button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e05b51c4-1181-43fb-88a1-5c482a50dedf');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Output words dataframe


```python
words_df
```





  <div id="df-68af9adb-8287-4e19-bf1e-db21634a178a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sentence</th>
      <td>What</td>
      <td>is</td>
      <td>the</td>
      <td>capital</td>
      <td>of</td>
      <td>the</td>
      <td>US</td>
    </tr>
    <tr>
      <th>gpt block 0</th>
      <td>the</td>
      <td>still</td>
      <td>same</td>
      <td>capital</td>
      <td>the</td>
      <td>same</td>
      <td>US</td>
    </tr>
    <tr>
      <th>gpt block 1</th>
      <td>the</td>
      <td>still</td>
      <td>same</td>
      <td>capital</td>
      <td>the</td>
      <td>same</td>
      <td>Army</td>
    </tr>
    <tr>
      <th>gpt block 2</th>
      <td>the</td>
      <td>really</td>
      <td>same</td>
      <td>capital</td>
      <td>the</td>
      <td>same</td>
      <td>Army</td>
    </tr>
    <tr>
      <th>gpt block 3</th>
      <td>the</td>
      <td>really</td>
      <td>difference</td>
      <td>capital</td>
      <td>the</td>
      <td>same</td>
      <td>National</td>
    </tr>
    <tr>
      <th>gpt block 4</th>
      <td>the</td>
      <td>really</td>
      <td>difference</td>
      <td>capital</td>
      <td>the</td>
      <td>world</td>
      <td>National</td>
    </tr>
    <tr>
      <th>gpt block 5</th>
      <td>the</td>
      <td>really</td>
      <td>difference</td>
      <td>capital</td>
      <td>the</td>
      <td>world</td>
      <td>National</td>
    </tr>
    <tr>
      <th>gpt block 6</th>
      <td>the</td>
      <td>really</td>
      <td>difference</td>
      <td>difference</td>
      <td>the</td>
      <td>world</td>
      <td>Embassy</td>
    </tr>
    <tr>
      <th>gpt block 7</th>
      <td>the</td>
      <td>happening</td>
      <td>best</td>
      <td>ization</td>
      <td>the</td>
      <td>world</td>
      <td>Dollar</td>
    </tr>
    <tr>
      <th>gpt block 8</th>
      <td>the</td>
      <td>the</td>
      <td>difference</td>
      <td>ization</td>
      <td>the</td>
      <td>world</td>
      <td>?</td>
    </tr>
    <tr>
      <th>gpt block 9</th>
      <td>the</td>
      <td>the</td>
      <td>difference</td>
      <td>ization</td>
      <td>the</td>
      <td>country</td>
      <td>?</td>
    </tr>
    <tr>
      <th>gpt block 10</th>
      <td>the</td>
      <td>the</td>
      <td>difference</td>
      <td>ization</td>
      <td>the</td>
      <td>world</td>
      <td>?</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-68af9adb-8287-4e19-bf1e-db21634a178a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-68af9adb-8287-4e19-bf1e-db21634a178a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-68af9adb-8287-4e19-bf1e-db21634a178a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Data visualization


```python
"""Final colored dataframe"""
fig, ax = plt.subplots(figsize=(25, 8))
ax = sns.heatmap(logits_df.astype(int), ax=ax, annot=words_df, 
                 cbar=True, fmt="", cmap=cc.kbgyw[::-1], 
                 linewidth=0.0, cbar_kws={'label': 'Logits'})
ax.set(title="Logit Lens Experiments", xlabel="Tokens", ylabel="Layer index",)
ax.xaxis.tick_bottom() 
ax.xaxis.set_label_position('bottom')
plt.show()
```


    
![png](logit_lens_files/logit_lens_25_0.png)
    


## Analysis of the results

In the explanation, we will use the term logits. The logits of a model are positive numbers representing the probability of belonging to the output class: the higher the logit, the more likely it is that the input will match the output. Logits are not probabilities because they can exceed 1. To pass from logits to probability it is necessary to apply the softmax function to logits for example.

The first salient information from the results is the inequality of confidence between the last layers and the first ones. Indeed, the first layers try to find correlated words while the last layer makes a decision. This gives an idea of the architecture of the gpt2 model.

The first column is also particular. The word "What" becomes "the". To understand this phenomenon we have to go back to the logits. The model is never sure what to put after. There is nothing in its memory, so it is hard to decide. So it decides to put a generalist word like "the".

Even though GPT-3 is presented as being much better than GPT-2, it already has some sentence comprehension skills. From "What is the", it confidently completes "difference" which is a likely follow-up. Then, from "What is the capital", it predicts that the sentence is "What is the capitalization" which is also a probable sentence. Finally, he understands that the sentence is a question because he predicts that the next word is a question mark. But it is clear that GPT-2 is far from the performance of GPT-3. If we write "I love hot chocolate with", GPT-2 predicts that the next word is "chocolate".
