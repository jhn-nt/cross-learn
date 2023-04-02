# cross-learn
## extensive scoring of crossvalidation loops.
![version](https://img.shields.io/badge/version-1.0-green)
![python](https://img.shields.io/badge/python->=3.8-green)

_cross-learn_ is an ensemble of sklearn wrappers aiming to simplify the validation of statistical learning models.  
Particularly, these libraries address how the _groups_ parameter is handled by scikit-learn, which has been bugging me for a while.   
The main features I focused on are:
* Cleanliness of code.
* Flexibility.
* Automation and completeness of models scoring.
* Simplification of nested crossvalidation procedures.

The code is functionally split in 3 separate modules: _crossvalidators_, _evaluation_ and _transformers_.

### _evaluation_ module
Contains the `crossvalidate_classification` and `crossvalidate_regression` methods, all-in-one wrappers to obtain crossvalidation and nested crossvalidation scores with any sklearn-like model or pipeline, but most importantly allows for _intra-fold dependencies_ during crossvalidation (ie nested crossvalidation with GroupKFold or similar).

Functionally, these methods act as simple scoring tracers to ease readability of evaluation metrics.


### _transformers_ module
Revisions of some vanilla sklearn transformers with some new functionality:  
* `DropColin`: Unsupervised filtering of linearly correlated features. 
* `DropColinCV`: Crossvalidated extension of `DropColin`. 
* `DropByMissingRate`: Filters out features missing more than a predefined thershold.
* `DropByMissingRateCV`: Crossvalidated extension of `DropByMissingRate`. 

## Installation Notes:

Run:  

```python
pip install "git+https://github.com/jhn-nt/cross-learn.git"
```


## Notes

These are libraires I have been developing during the years on personal projects.
After noticing I was re-writing time after time the same routines for the same problems I have decided to write them one last time for good.  
Hopefully they will be of good use for others as well.


The code is fully scikit-learn compatbile and likely will see major revisions as I come up with new ideas.
I have been moslty focusing on polishness and ease-of-use with a great focus on typing.

Most of all, writing these libraries has been a fantastic exercise to learn to build a cleaner and more re-usable code.

Very open to any feedback

Cheers!