.PHONY: all clean

.SECONDARY: 

include Makefile.local
include Makefile.list

BASE=~/projs/VT/experiments/e130117_iccv
PLOT_SYNTHETIC=$(BASE)/plot_synthetic.py 
FEATURES=$(BASE)/features.py
CL_TRAIN=$(BASE)/classifier_train.py
CL_PREDICT=$(BASE)/classifier_predict.py
EVALUATE=$(BASE)/evaluate.py
DISSIM=$(BASE)/dissimilarities.py --jobs=$(N_JOBS)
PLOT_WEIGHTS=$(BASE)/plot_weights.py
PR_TRAIN=$(BASE)/propagate_train.py --jobs=$(N_JOBS)
PLOT_CV=$(BASE)/plot_propagator_cv_results.py
PR_PREDICT=$(BASE)/propagate_predict.py
SUBSAMPLE=$(BASE)/subsample.py -n20
CONVERT_TRUTH=$(BASE)/convert.py --truth=stage --no-timestamp 
CONVERT=$(BASE)/convert.py --no-timestamp 
LOCAL_COPY=$(BASE)/copy_images.py
PLOT_WEIGHTS=$(BASE)/plot_weights.py
DISSIM_THRESH=$(BASE)/dissimilarities_thresh.py
VIZU_KNN=$(BASE)/vizu_knn.py

PR_TRAIN_DISSIMS:=$(PR_TRAIN_DATA:%=%-data-dissim.csv)
PR_TRAIN_TRUTH:=$(PR_TRAIN_DATA:%=%-truth.csv)
PR_TRAIN_PREDS:=$(PR_TRAIN_DATA:%=%-data-predictions.csv)
PR_TRAIN_PROPS:=$(PR_TRAIN_DATA:%=%-data-propagated.csv)

all: predictions-cm.svg propagator-cv.svg propagated-cm.svg plot-weights

clean:
	rm -f classifier.dat propagator.dat 2>/dev/null
	rm -f lp.dat chaincodes.dat dictionary.dat 2>/dev/null
	rm -f *.svg 2>/dev/null
	rm -f *-predictions.csv *-propagated.csv *-feats.csv *-truth.csv *-data.csv *-dissim.csv *-distances.dat 2>/dev/null

local-images: $(PR_TRAIN_DATA:%=%-data.csv)
	$(LOCAL_COPY) -r ~/vt_project/Segmentation/Fine/MetaSys/ -o ../images/ $^
	ln -s ../images/ image

truths: $(PR_TRAIN_TRUTH)
feats: $(PR_TRAIN_DATA:%=%-truth-feats.csv) $(PR_TRAIN_DATA:%=%-data-feats.csv)
predictions: $(PR_TRAIN_PREDS)
propagated: $(PR_TRAIN_PROPS)
data: $(PR_TRAIN_DATA:%=%-data.csv)
weights: $(PR_TRAIN_DATA:%=%-data-dissim.csv)
plot-weights: $(PR_TRAIN_DATA:%=%-data-dissim.svg)

threshold-weights: $(PR_TRAIN_DATA:%=%-data-dissim.csv)
	for file in $^; do $(DISSIM_THRESH) -m knn -d $$file; done

SpecificStages-truth.csv: truth/SpecificStages*.csv
	$(CONVERT_TRUTH) --no-timestamp -p $(IMAGE_PREFIX) -o . 'truth/SpecificStages*.csv'
	mv truth.csv $@

Sampled-ek_%-truth.csv: truth/Sampled-ek_%.csv
	$(CONVERT_TRUTH) -p $(IMAGE_PREFIX) -o . --outname-prefix=Sampled-ek_ $<

Sampled-ek_%-data.csv: all/Sampled-ek_%.csv
	$(CONVERT) -p $(IMAGE_PREFIX) -o . --outname-prefix=Sampled-ek_ $<

Sampled-fi_%-truth.csv: truth/Sampled-fi_%.csv
	$(CONVERT_TRUTH) -p $(IMAGE_PREFIX) -o . --outname-prefix=Sampled-fi_ $<

Sampled-fi_%-data.csv: all/Sampled-fi_%.csv
	$(CONVERT) -p $(IMAGE_PREFIX) -o . --outname-prefix=Sampled-fi_ $<

Sampled-ts_%-truth.csv: truth/Sampled-ts_%.csv
	$(CONVERT_TRUTH) -p $(IMAGE_PREFIX) -o . --outname-prefix=Sampled-ts_ $<

Sampled-ts_%-data.csv: all/Sampled-ts_%.csv
	$(CONVERT) -p $(IMAGE_PREFIX) -o . --outname-prefix=Sampled-ts_ $<

%-feats.csv: %.csv chaincode.args
	$(FEATURES) -m chaincode -a chaincode.args -l $< -o .

classifier.dat: $(CL_TRAIN_DATA)-truth.csv $(CL_TRAIN_DATA)-truth-feats.csv svm.args
	$(CL_TRAIN) -m svm -a svm.args -f $(CL_TRAIN_DATA)-truth-feats.csv -t $(CL_TRAIN_DATA)-truth.csv -o .

%-predictions.csv: %.csv classifier.dat
	$(CL_PREDICT) -l $< -m classifier.dat -o .

predictions-cm.svg: $(PR_TRAIN_PREDS) $(PR_TRAIN_TRUTH)
	$(EVALUATE) --all-prefix=predictions -p $(PR_TRAIN_PREDS) -t $(PR_TRAIN_TRUTH) -o .

%-data-predictions-cm.svg: %-data-predictions.csv %-truth.csv
	$(EVALUATE) -p $*-data-predictions.csv -t $*-truth.csv -o .

%-data-dissim.csv: %-data.csv dissim.args %-truth.csv
	$(DISSIM) -m expression -a dissim.args -l $< -o .

%-data-dissim.svg: %-data-dissim.csv %-truth.csv
	$(PLOT_WEIGHTS) -w $< -o . -t $*-truth.csv

propagator.dat: $(PR_TRAIN_DISSIMS) $(PR_TRAIN_TRUTH) $(PR_TRAIN_PREDS) harmonic.args
	$(PR_TRAIN) -m harmonic -a harmonic.args -t $(PR_TRAIN_TRUTH) -d $(PR_TRAIN_DISSIMS) -p $(PR_TRAIN_PREDS) -o .

propagator-cv.svg: propagator.dat
	$(PLOT_CV) -m propagator.dat -o .

%-propagated.csv: %-dissim.csv %-predictions.csv propagator.dat
	$(PR_PREDICT) -m propagator.dat -d $*-dissim.csv -p $*-predictions.csv -o .

propagated-cm.svg: $(PR_TRAIN_PROPS) $(PR_TRAIN_TRUTH)
	$(EVALUATE) --all-prefix=propagated -p $(PR_TRAIN_PROPS) -t $(PR_TRAIN_TRUTH) -o .

%-data-propagated-cm.svg: %-data-propagated.csv %-truth.csv
	$(EVALUATE) -p $*-data-propagated.csv -t $*-truth.csv -o .

%-data-neighbors.html: %-data.csv %-data-dissim.csv %-truth.csv %-data-predictions.csv %-data-propagated.csv
	$(VIZU_KNN) -l $*-data.csv -d $*-data-dissim.csv -t $*-truth.csv -p $*-data-predictions.csv -q $*-data-propagated.csv -o .

errors: $(PR_TRAIN_DATA:%=%-data-errors.html)

%-data-errors.html: %-data.csv %-data-dissim.csv %-truth.csv %-data-predictions.csv %-data-propagated.csv
	$(VIZU_KNN) --only-errors -l $*-data.csv -d $*-data-dissim.csv -t $*-truth.csv -p $*-data-predictions.csv -q $*-data-propagated.csv -o .

errors-neighbors.html:
	$(VIZU_KNN) --only-errors -l $(PR_TRAIN_DATA:%=%-data.csv) -d $(PR_TRAIN_DISSIMS) -t $(PR_TRAIN_TRUTH) -p $(PR_TRAIN_PREDS) -q $(PR_TRAIN_PROPS) -o .
