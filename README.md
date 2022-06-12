# Transformer-Text-AutoEncoder
Transformer Text AutoEncoder: An autoencoder is a type of artificial neural network used to learn efficient encodings of unlabeled data, the same is employed for textual data employing pre-trained models from the hugging-face library.

## Installation:

```console
pip install Transformer-Text-AutoEncoder
```

## Execution:

```py
ttae = TTAE('./data/FinancialNews.txt', no_cuda)
ttae.train(10, batch_size=1)
print(ttae.predict("Hello world"))
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
