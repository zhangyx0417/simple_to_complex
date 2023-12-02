# From Simple to Complex: A Progressive Framework for Document-level Informative Argument Extraction

Code for our EMNLP 2023 paper.

## Model Overview
The figure below illustrates our simple-to-complex progressive framework for document-level informative argument extraction. First, we calculate the difficulty of each event in a document $D$ and obtain a new prediction order for that event. Second, we reorder events in $D$ from simple to complex, and predict them accordingly. S2C denotes Simple-to-Complex, while F2B denotes Front-to-Back. Here, we plot the process of predicting the arguments of $E_2$.

![Model](figs/model.png)


## Dependencies 
- pytorch=1.8.0
- transformers=3.1.0
- pytorch-lightning=1.0.6
- spacy=3.0
- sentence-transformers=2.1.0


## Datasets
- WikiEvents (download from [this repo](https://github.com/xinyadu/memory_docie))


## Running

- Confidence calibration
	
	```
	bash scripts/calibrate.sh
	```

- Training

	```
	bash scripts/train_kairos.sh
	```

- Testing

	```
	bash scripts/test_kairos.sh
	```


## Citation
If you find our work useful, please cite as follows:

```
@inproceedings{
huang2023from,
title={From Simple to Complex: A Progressive Framework for Document-level Informative Argument Extraction},
author={Quzhe Huang and Yanxi Zhang and Dongyan Zhao},
booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
year={2023},
url={https://openreview.net/forum?id=dp9jTeKXec}
}
```
