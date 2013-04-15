#Label propagation pipeline
The pipeline consists of two blocks, a classification based on single samples, and a label propagation, which uses a similarity between the samples to denoise the labels assigned by the classifier. So the classifier provides unaries whereas the similarities provide the pairwise terms.

## Data files

### List file and truth file

Input data is specified in listfiles as CSV tables which have at least one column called ID.
Minimal file should look like this:

	ID
	001
	002
	003
	...

Meta information can be contained inside a header commented by `#` signs and encoded in YAML format.
The interpretation of the individual CSV columns is entirely up to the different scripts, so a file with one image per sample can look like this:

	#path_prefix=/an/optional/path/prefix
	ID	path
	001	somedir/file001.png
	002 dir2/file002.png
	003 anotherdir/file003.png

The file can contain any number of other columns. A truth file is just a list file with an additional _integer_ column identified by `truth` item in meta data. Labels are stored in meta data as well:

	#basedir=/an/optional/path/prefix
	#truth_labels={ 1: 'classA', 2: 'classB' }
	#truth=class
	ID	path	class
	001	somedir/file001.png	1
	002 dir2/file002.png	2
	003 anotherdir/file003.png	1

### Feature file and weights file

A feature file is a CSV table of the following form:

	#methodname=compute_features_method
	#argsfile=argsfile
	#additional_optional_data1=...(eg. a dictionary filename)
	ID	f1	f2	f3	...
	001 2.	.4	-1	...
	002	3.  .5  .1	...
	003 2.	.7	-2	...

Column `ID` should match with the corresponding list file, all other columns are considered to be features.

A weights file is a feature file with a square matrix of weights saved as features and indexed by sample IDs:

	#methodname=...
	ID	001	002	003	...
	001 2.	.4	-1	...
	002	3.  .5  .1	...
	003 2.	.7	-2	...

### Classifier file

A pickled (serialized) bzip2-compressed Python dictionary with the following entries:

	features = { 'methodname': ..., 'argsfile': ... }
	meta = ...
	truth = ...
	classifier = classifier_object

### Prediction file

A prediction file is a truth file with additional columns:

- `pred` containing the predicted class, and
- `probN` containing the posterior probability of the class `N`.

### Propagator file

A pickled (serialized) bzip2-compressed Python dictionary with the following entries:

    propagator = { propagator_params... }
    meta = ...

## Classification

* `features.py` - computes features from input data
* `classifier_train.py` - trains a classifier
* `classifier_predict.py` - predicts labels using an existing classifier file

### features.py

Computes features for all the input data. It contains a function `compute_features(method_name, method_args, data)` which based on `method_name`  calls the specific method for each sample in data:

	features = []
	for sample in data:
		features += method_table[method_name](sample, cache=cache, **method_args)
	return features, feature_names

Command-line interface:

	features.py -m method_name -a argsfile -l listfile

This runs feature extraction and saves output features for data listed specified in the listfiles into `features.csv`.

### classifier_train.py

Runs feature extraction, selects optimal hyper-parameters by cross-validation, learns the final model from all the data, and finally saves the model. It contains a function `train_classifier(features, truth)`. Prints out training evaluation.

Command-line interface:

	train.py -m svm -t truthfile -f featurefile

where truth file is a listfile with columns ID, path, and truth.

It creates a file `classifier.dat`.

### classifier_predict.py

Loads an existing classifier and applies it to the data. Outputs predictions.

Command-line interface:

	predict.py -l listfile -m classifierfile

It creates a prediction file `pred.csv`.

## Label propagation

* `dissimilarities.py` - computes dissimilarities of sample pairs
* `propagate_train.py` - finds optimal hyper-parameters of label propagation
* `propagate_predict.py` - propagates labels from a classifier using weights
* `evaluate.py` - evaluates propagated labels using ground truth


### weights.py

Evaluates mutual similarities between all the sample pairs. Defines a function `compute_dissimilarities(method_name, method_args, data)`.

Command-line interface:

	dissimilarities.py -m method_name -a argsfile -l listfile

It creates a weights file `dissim.csv`.

A `threshold` item in the arguments files defines how are the similarities thresholded. These similarities are used to setup edge weights for the label propagation, so thresholding makes the used graph sparse.

### propagate_train.py

Searches for optimal hyper-parameters defined in an arguments file. This having at least partial truth for a set of listfiles, predictions from an already learned classifier on the whole listfiles, and dissimilarities for all the listfiles. Defines a function `train(method_name, method_args, data)`.

Command-line interface:

	propagate-train.py -m method_name -p predfile1 predfile2 ... -d dissimfile1 dissimfile2 ... -t truthfile1 truthfile2 ...

Arguments file can look like this:

	hyper_params: [ unary_weight, bandwidth ]
	scoring: accuracy
	unary_weight: [ 0.25, 0.5, 1., 2., 4., 8. ]
	bandwidth: [ .5, 1., 5., 10., 50. ]

### propagate_predict.py

Propagates labels according to predictions for different samples and according to the edge weights computed by `dissimilarities.py`. Defines a function `propagate(method_name, method_args, predictions, dissim)`.

Command-line interface:

	propagate.py -m model -p predictionfile -d dissimfile

It creates a prediction file `prop.csv`.

### evaluate.py

Evaluates the provided labels against the truthfile.

Command-line interface:

	test.py -t truthfile -p predictionfile

It creates various output SVG, and HTML files.
