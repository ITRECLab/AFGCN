# AFGCN


## Updates

## Requirements

* Python 3.6
* PyTorch 1.0.0
* SpaCy 2.0.18
* numpy 1.15.4

## Usage

* Install [SpaCy](https://spacy.io/) package and language models with
```bash
pip install spacy
```
and
```bash
python -m spacy download en
```
* Generate graph data with
```bash
python dependency_graph.py
python dependency_dist.py
```
* Download pretrained GloVe embeddings with this [link](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and extract `glove.840B.300d.txt` into `glove/`.
* Train with command, optional arguments could be found in [train.py](/train.py)
```bash
python train.py --model_name asgcn --dataset rest14 --save True
```
* Infer with [infer.py](/infer.py)

## Model



## Citation

If you use the code in your paper, please kindly star this repo and cite our paper



## Credits

* Code of this repo heavily relies on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch), in which I am one of the contributors.
* For any issues or suggestions about this work, don't hesitate to create an issue or directly contact me via [gene_zhangchen@163.com](mailto:gene_zhangchen@163.com) !