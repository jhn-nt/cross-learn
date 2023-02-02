# JHN_AI
## General purpose ML dev tools.

Libraires I have been developing during the years on personal projects.
After noticing I was re-writing time after time the same routines for the same problems I have decided to write them one last time for good.  
Hopefully they will be of good use for others as well.


The code is scikit-learn compatbile (mostly) and likely will see major revisions as I come up with new ideas.
I have been moslty focusing on polishness and ease-of-use with a great focus on typing.

Most of all, writing these libraries has been a fantastic exercise to learn to build a cleaner and more re-usable code.

Very open to any feedback

Cheers!

## Overview
_jhn_ai_ are an ensemble of sklearn wrappers aiming to simplify the validation of supervised machine learning models.  
Particularly, these libraries address how the _groups_ parameter is handled by scikit-learn, which has been bugging me for a while.   
The main features I focused on are:
* Cleanliness of code.
* Flexibility.
* Automation and completeness of models scoring.
* Simplification of nested crossvalidation procedures.

The code is functionally split in 3 separate modules: _crossvalidators_, _evaluation_ and _transformers_.

### _evaluation_ module
Contains the ```supervised_crossvalidation``` method, an all-in-one wrapper to obtain crossvalidation and nested crossvalidation scores with any sklearn-like model or pipeline, but most importantly allows for _intra-fold dependencies_ during crossvalidation (ie nested crossvalidation with GroupKFold or similar).

```find_inertia_elbow``` is another handy method for unsupervised inertia based clustering models (aka KMeans etc etc). It allows for quick identification of the inertia elbo using the triangle method.

### _crossvalidators_ module
The ```crossvalidators``` module introduces two new crossvalidators:
* ```WalkForwardCV```: Similar to sklearn TimeSeriesSplit in concept, but extended to support for group-wise time splitting. 
* ```StackedCV```: Combines the effect of different validation models. For example, if you need to validate data of different costumers with different purchase habits, you can combine GroupKFold and StratifiedKFold within ```StackedCV```. 

### _transformers_ module
Revisions of some vanilla sklearn transformers with some new functionality:  
* ```RateImputer```: Drop columns missing more than user-predifined rate and imputes the rest. 
* ```ColinearityRemover```: An unsupervised feature filtering model which removes correlated features.
* ```IsolationForest```: sklearn IsolationForest made compatible with pipelines and crossvalidation routines.

## Installation Notes:

Run:  

```pip install "git+https://github.com/jhn-nt/jhn_ai.git"```
