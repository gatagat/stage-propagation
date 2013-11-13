.PHONY: eval-prop eval-pred

BASE=~/projs/propagation/
GENERATE=$(BASE)generate.py
PLOT_SYNTHETIC=$(BASE)plot_synthetic.py
FEATURES=$(BASE)features.py
CL_TRAIN=$(BASE)classifier_train.py
CL_PREDICT=$(BASE)classifier_predict.py
EVALUATE=$(BASE)evaluate.py
DISSIM=$(BASE)dissimilarities.py
PLOT_WEIGHTS=$(BASE)plot_weights.py
PR_TRAIN=$(BASE)propagate_train.py
PLOT_CV=$(BASE)plot_propagator_cv_results.py
PR_PREDICT=$(BASE)propagate_predict.py
SUBSAMPLE=$(BASE)subsample.py -n20

# SEEDS
# train-classifier.csv.gz: 1448504125468647
# train-propagator.csv.gz: 1448504125479129
# test.csv.gz: 1448504125483280

all: eval-pred eval-prop

clean:
	rm -f train-classifier*.csv.gz train-propagator*.csv.gz test*.csv.gz 2>/dev/null
	rm -f classifier.dat propagator.dat 2>/dev/null
	rm -fr train-classifier train-propagator test 2>/dev/null

eval-pred: train-propagator/train-propagator-predictions-cm.svg
eval-prop: test/test-predictions-cm.svg test/test-propagated-cm.svg \
	train-propagator/propagator-cv.svg train-propagator/train-propagator-propagated-cm.svg

train-classifier.csv.gz: generate.args
	$(GENERATE) -a generate.args $@
	$(PLOT_SYNTHETIC) -x f00 -y f01 -t $@ -o train-classifier

train-classifier-part.csv: train-classifier.csv.gz
	$(SUBSAMPLE) -l $^ -o .

train-propagator.csv.gz: generate.args
	$(GENERATE) -a generate.args $@
	$(PLOT_SYNTHETIC) -x f00 -y f01 -t $@ -o train-propagator

train-propagator-part.csv: train-propagator.csv.gz
	$(SUBSAMPLE) -l $^ -o .

test.csv.gz: generate.args
	$(GENERATE) -a generate.args $@
	$(PLOT_SYNTHETIC) -x f00 -y f01 -t $@ -o test

test-part.csv.gz: test.csv.gz
	$(SUBSAMPLE) -l $^ -o .

train-classifier-feats.csv.gz: train-classifier.csv.gz features.args
	$(FEATURES) -m pregenerated -a features.args -l train-classifier.csv.gz -o .

classifier.dat: train-classifier.csv.gz train-classifier-feats.csv.gz
	$(CL_TRAIN) -m svm -a svm.args -f train-classifier-feats.csv.gz -t train-classifier.csv.gz -o .

train-propagator-predictions.csv.gz: classifier.dat train-propagator.csv.gz
	$(CL_PREDICT) -l train-propagator.csv.gz -m classifier.dat -o .

test-predictions.csv.gz: classifier.dat test.csv.gz
	$(CL_PREDICT) -l test.csv.gz -m classifier.dat -o .

train-propagator/train-propagator-predictions-cm.svg: train-propagator-predictions.csv.gz train-propagator.csv.gz train-propagator-part.csv
	$(PLOT_SYNTHETIC) -x f00 -y f01 -l train-propagator.csv.gz -p train-propagator-predictions.csv.gz -o train-propagator
	$(EVALUATE) -p train-propagator-predictions.csv.gz -t train-propagator.csv.gz -o train-propagator
	$(EVALUATE) -p train-propagator-predictions.csv.gz -t train-propagator-part.csv -o train-propagator-part

train-propagator-dissim.csv.gz: dissim.args train-propagator.csv.gz
	$(DISSIM) -m pregenerated -a dissim.args -l train-propagator.csv.gz -o .
	$(PLOT_WEIGHTS) -w $@ -o train-propagator -t train-propagator.csv.gz

test-dissim.csv.gz: dissim.args test.csv.gz
	$(DISSIM) -m pregenerated -a dissim.args -l test.csv.gz -o .
	$(PLOT_WEIGHTS) -w $@ -o test -t test.csv.gz

propagator.dat: train-propagator-part.csv train-propagator-dissim.csv.gz train-propagator-predictions.csv.gz harmonic.args
	$(PR_TRAIN) -m harmonic -a harmonic.args -t train-propagator-part.csv -d train-propagator-dissim.csv.gz -p train-propagator-predictions.csv.gz -o .

train-propagator-propagated.csv.gz: train-propagator-dissim.csv.gz train-propagator-predictions.csv.gz propagator.dat
	$(PR_PREDICT) -m propagator.dat -d train-propagator-dissim.csv.gz -p train-propagator-predictions.csv.gz -o .

train-propagator/train-propagator-propagated-cm.svg: train-propagator-propagated.csv.gz train-propagator.csv.gz train-propagator-part.csv
	$(PLOT_SYNTHETIC) -x f00 -y f01 -l train-propagator.csv.gz -p train-propagator-propagated.csv.gz -o train-propagator
	$(EVALUATE) -p train-propagator-propagated.csv.gz -t train-propagator.csv.gz -o train-propagator
	$(EVALUATE) -p train-propagator-propagated.csv.gz -t train-propagator-part.csv -o train-propagator-part

train-propagator/propagator-cv.svg: propagator.dat
	$(PLOT_CV) -m propagator.dat -o train-propagator

test-propagated.csv.gz: test-dissim.csv.gz test-predictions.csv.gz propagator.dat
	$(PR_PREDICT) -m propagator.dat -d test-dissim.csv.gz -p test-predictions.csv.gz -o .

test/test-predictions-cm.svg: test-predictions.csv.gz test.csv.gz test-part.csv.gz
	$(PLOT_SYNTHETIC) -x f00 -y f01 -l test.csv.gz -p test-predictions.csv.gz -o test
	$(EVALUATE) -p test-predictions.csv.gz -t test.csv.gz -o test
	$(EVALUATE) -p test-predictions.csv.gz -t test-part.csv -o test-part

test/test-propagated-cm.svg: test-propagated.csv.gz test.csv.gz test-part.csv.gz
	$(PLOT_SYNTHETIC) -x f00 -y f01 -l test.csv.gz -p test-propagated.csv.gz -o test
	$(EVALUATE) -p test-propagated.csv.gz -t test.csv.gz -o test
	$(EVALUATE) -p test-propagated.csv.gz -t test-part.csv -o test-part

