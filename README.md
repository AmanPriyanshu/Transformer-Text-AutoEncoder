# Transformer-Text-AutoEncoder
Transformer Text AutoEncoder: An autoencoder is a type of artificial neural network used to learn efficient encodings of unlabeled data, the same is employed for textual data employing pre-trained models from the hugging-face library.

## Installation:

```console
pip install Transformer-Text-AutoEncoder
```

## Execution:

```py
from Transformer_Text_AutoEncoder.AutoEncoder import TTAE

def read(path='./data/FinancialNews.txt'):
  with open(path, "r", encoding='utf-8', errors='ignore') as f:
    data = [i.strip() for i in f.readlines()]
  return data

sentences = read()
print(sentences[:3])
ttae = TTAE(sentences)
ttae.train(10, batch_size=1)
print(ttae.predict("According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing ."))
```

returns the `predicted sentence` as well as the `embeddings`.

## Cite Work:

```console
@inproceedings{ttae,
  title = {Transformer-Text-AutoEncoder},
  author = {Aman Priyanshu},
  year = {2022},
  publisher = {{GitHub}},
  url = {https://github.com/AmanPriyanshu/Transformer-Text-AutoEncoder/}
}
```
