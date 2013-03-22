.PHONY: eval-prop eval-pred

BASE=~/projs/VT/experiments/e130117_iccv
GENERATE=$(BASE)/generate.py
PLOT_SYNTHETIC=$(BASE)/plot_synthetic.py 
FEATURES=$(BASE)/features.py
CL_TRAIN=$(BASE)/classifier_train.py
CL_PREDICT=$(BASE)/classifier_predict.py
EVALUATE=$(BASE)/evaluate.py
DISSIM=$(BASE)/dissimilarities.py
PLOT_WEIGHTS=$(BASE)/plot_weights.py
PR_TRAIN=$(BASE)/propagate_train.py
PLOT_CV=$(BASE)/plot_propagator_cv_results.py
PR_PREDICT=$(BASE)/propagate_predict.py
SUBSAMPLE=$(BASE)/subsample.py -n20

all: eval-pred eval-prop

clean:
	rm -f train-classifier*.csv train-propagator*.csv test*.csv 2>/dev/null
	rm -f classifier.dat propagator.dat 2>/dev/null
	rm -fr train-classifier train-propagator test 2>/dev/null

eval-pred: train-propagator/train-propagator-predictions-cm.svg
eval-prop: test/test-predictions-cm.svg test/test-propagated-cm.svg train-propagator/propagator-cv.svg train-propagator/train-propagator-propagated-cm.svg

train-classifier.csv: generate.args
	$(GENERATE) -a generate.args $@
	$(PLOT_SYNTHETIC) -x f00 -y f01 -t $@ -o train-classifier

train-classifier-part.csv: train-classifier.csv
	$(SUBSAMPLE) -l $^ -o .

train-propagator.csv: generate.args
	$(GENERATE) -a generate.args $@
	$(PLOT_SYNTHETIC) -x f00 -y f01 -t $@ -o train-propagator

train-propagator-part.csv: train-propagator.csv
	$(SUBSAMPLE) -l $^ -o .

test.csv: generate.args
	$(GENERATE) -a generate.args $@
	$(PLOT_SYNTHETIC) -x f00 -y f01 -t $@ -o test

test-part.csv: test.csv
	$(SUBSAMPLE) -l $^ -o .

train-classifier-feats.csv: train-classifier.csv features.args
	$(FEATURES) -m pregenerated -a features.args -l train-classifier.csv -o .

classifier.dat: train-classifier.csv train-classifier-feats.csv
	$(CL_TRAIN) -m svm -a svm.args -f train-classifier-feats.csv -t train-classifier.csv -o .

train-propagator-predictions.csv: classifier.dat train-propagator.csv
	$(CL_PREDICT) -l train-propagator.csv -m classifier.dat -o .

test-predictions.csv: classifier.dat test.csv
	$(CL_PREDICT) -l test.csv -m classifier.dat -o .

train-propagator/train-propagator-predictions-cm.svg: train-propagator-predictions.csv train-propagator.csv train-propagator-part.csv
	$(PLOT_SYNTHETIC) -x f00 -y f01 -l train-propagator.csv -p train-propagator-predictions.csv -o train-propagator
	$(EVALUATE) -p train-propagator-predictions.csv -t train-propagator.csv -o train-propagator
	$(EVALUATE) -p train-propagator-predictions.csv -t train-propagator-part.csv -o train-propagator-part

train-propagator-dissim.csv: dissim.args train-propagator.csv
	$(DISSIM) -m pregenerated -a dissim.args -l train-propagator.csv -o .
	$(PLOT_WEIGHTS) -w $@ -o train-propagator -t train-propagator.csv

test-dissim.csv: dissim.args test.csv
	$(DISSIM) -m pregenerated -a dissim.args -l test.csv -o .
	$(PLOT_WEIGHTS) -w $@ -o test -t test.csv

propagator.dat: train-propagator-part.csv train-propagator-dissim.csv train-propagator-predictions.csv harmonic.args
	$(PR_TRAIN) -m harmonic -a harmonic.args -t train-propagator-part.csv -d train-propagator-dissim.csv -p train-propagator-predictions.csv -o .

train-propagator-propagated.csv: train-propagator-dissim.csv train-propagator-predictions.csv propagator.dat
	$(PR_PREDICT) -m propagator.dat -d train-propagator-dissim.csv -p train-propagator-predictions.csv -o .

train-propagator/train-propagator-propagated-cm.svg: train-propagator-propagated.csv train-propagator.csv train-propagator-part.csv
	$(PLOT_SYNTHETIC) -x f00 -y f01 -l train-propagator.csv -p train-propagator-propagated.csv -o train-propagator
	$(EVALUATE) -p train-propagator-propagated.csv -t train-propagator.csv -o train-propagator
	$(EVALUATE) -p train-propagator-propagated.csv -t train-propagator-part.csv -o train-propagator-part

train-propagator/propagator-cv.svg: propagator.dat
	$(PLOT_CV) -m propagator.dat -o train-propagator

test-propagated.csv: test-dissim.csv test-predictions.csv propagator.dat
	$(PR_PREDICT) -m propagator.dat -d test-dissim.csv -p test-predictions.csv -o .

test/test-predictions-cm.svg: test-predictions.csv test.csv
	$(PLOT_SYNTHETIC) -x f00 -y f01 -l test.csv -p test-predictions.csv -o test
	$(EVALUATE) -p test-predictions.csv -t test.csv -o test
	$(EVALUATE) -p test-predictions.csv -t test-part.csv -o test-part

test/test-propagated-cm.svg: test-propagated.csv test.csv test-part.csv
	$(PLOT_SYNTHETIC) -x f00 -y f01 -l test.csv -p test-propagated.csv -o test
	$(EVALUATE) -p test-propagated.csv -t test.csv -o test
	$(EVALUATE) -p test-propagated.csv -t test-part.csv -o test-part

