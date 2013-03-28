.PHONY: all clean

.SECONDARY: 

BASE=~/projs/VT/experiments/e130117_iccv
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
CONVERT_TRUTH=$(BASE)/convert.py --truth=stage --no-timestamp 
CONVERT=$(BASE)/convert.py --no-timestamp 

CL_TRAIN_DATA=Test_20120907_037509~H
PR_TRAIN_DATA=\
20120604_9675 \
20120624_027321~A \
20120630_027678~A \
20120630_027684~A \
20120712_039525 \
20120719_1082~A \
20120722_034802~A \
20120726_019235~A \
20120726_022015 \
20120813_6477 \
Test_20120907_013456~B \
Test_20120907_1787~H
PR_TRAIN_DISSIMS:=$(PR_TRAIN_DATA:%=%-data-dissim.csv)
PR_TRAIN_TRUTH:=$(PR_TRAIN_DATA:%=%-truth.csv)
PR_TRAIN_PREDS:=$(PR_TRAIN_DATA:%=%-data-predictions.csv)
CL_CMS:=$(PR_TRAIN_DATA:%=%-data-predictions-cm.svg)
PR_CMS:=$(PR_TRAIN_DATA:%=%-data-propagated-cm.svg)

all: $(CL_CMS) $(PR_CMS) propagator-cv.svg

clean:
	rm -f classifier.dat propagator.dat 2>/dev/null
	rm -f lp.dat chaincodes.dat dictionary.dat 2>/dev/null
	rm -f *.svg 2>/dev/null
	rm -f *-predictions.csv *-propagated.csv *-feats.csv *-truth.csv *-data.csv *-dissim.csv *-distances.dat 2>/dev/null

%-truth.csv: truth/%.csv
	$(CONVERT_TRUTH) -p /Users/kazmar/tmp/Segmentation/Fine/MetaSys/ -o . $<

%-data.csv: all/%.csv
	$(CONVERT) -p /Users/kazmar/tmp/Segmentation/Fine/MetaSys/ -o . $<

%-feats.csv: %.csv chaincode.args
	$(FEATURES) -m chaincode -a chaincode.args -l $< -o .

classifier.dat: $(CL_TRAIN_DATA)-truth.csv $(CL_TRAIN_DATA)-truth-feats.csv svm.args
	$(CL_TRAIN) -m svm -a svm.args -f $(CL_TRAIN_DATA)-truth-feats.csv -t $(CL_TRAIN_DATA)-truth.csv -o .

%-predictions.csv: %.csv classifier.dat
	$(CL_PREDICT) -l $< -m classifier.dat -o .

%-data-predictions-cm.svg: %-data-predictions.csv %-truth.csv
	$(EVALUATE) --all-prefix=predictions -p $*-data-predictions.csv -t $*-truth.csv -o .

%-data-dissim.csv: %-data.csv ncclog.args %-truth.csv
	$(DISSIM) -m expression -a ncclog.args -l $< -o .
	$(PLOT_WEIGHTS) -w $@ -o . -t $*-truth.csv

propagator.dat: $(PR_TRAIN_DISSIMS) $(PR_TRAIN_TRUTH) $(PR_TRAIN_PREDS) harmonic.args
	$(PR_TRAIN) -m harmonic -a harmonic.args -t $(PR_TRAIN_TRUTH) -d $(PR_TRAIN_DISSIMS) -p $(PR_TRAIN_PREDS) -o .

propagator-cv.svg: propagator.dat
	$(PLOT_CV) -m propagator.dat -o .

%-propagated.csv: %-dissim.csv %-predictions.csv propagator.dat
	$(PR_PREDICT) -m propagator.dat -d $*-dissim.csv -p $*-predictions.csv -o .

%-data-propagated-cm.svg: %-data-propagated.csv %-truth.csv
	$(EVALUATE) --all-prefix=propagated -p $*-data-propagated.csv -t $*-truth.csv -o .

