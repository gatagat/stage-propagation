.PHONY: all clean cl-train pr-train test local-images

.SECONDARY: 

include Makefile.local
include Makefile.list

SRC_BASE=../../
PLOT_SYNTHETIC=$(SRC_BASE)plot_synthetic.py
FEATURES=$(SRC_BASE)features.py
CL_TRAIN=$(SRC_BASE)classifier_train.py --jobs=$(N_JOBS)
CL_PREDICT=$(SRC_BASE)classifier_predict.py
EVALUATE=$(SRC_BASE)evaluate.py
DISSIM=$(SRC_BASE)dissimilarities.py --jobs=$(N_JOBS)
PLOT_WEIGHTS=$(SRC_BASE)plot_weights.py
PR_TRAIN=$(SRC_BASE)propagate_train.py --jobs=$(N_JOBS)
PLOT_CV=$(SRC_BASE)plot_propagator_cv_results.py
PR_PREDICT=$(SRC_BASE)propagate_predict.py
SUBSAMPLE=$(SRC_BASE)subsample.py -n20
CONVERT_TRUTH=$(SRC_BASE)convert.py --truth=stage --no-timestamp
CONVERT=$(SRC_BASE)convert.py --no-timestamp
LOCAL_COPY=$(SRC_BASE)copy_images.py
PLOT_WEIGHTS=$(SRC_BASE)plot_weights.py
VIZU_KNN=$(SRC_BASE)vizu_knn.py

PR_TRAIN_DATAS:=$(PR_TRAIN_DATA:%=%-data.csv)
PR_TRAIN_DISSIMS:=$(PR_TRAIN_DATA:%=%-data-dissim.csv.gz)
PR_TRAIN_TRUTHS:=$(PR_TRAIN_DATA:%=%-truth.csv)
PR_TRAIN_PREDS:=$(PR_TRAIN_DATA:%=%-data-predictions.csv.gz)
PR_TRAIN_PROPS:=$(PR_TRAIN_DATA:%=%-data-propagated.csv.gz)

PR_TEST_TRUTHS:=$(PR_TEST_DATA:%=%-truth.csv)
PR_TEST_PREDS:=$(PR_TEST_DATA:%=%-data-predictions.csv.gz)
PR_TEST_PROPS:=$(PR_TEST_DATA:%=%-data-propagated.csv.gz)

all: predictions-cm.svg propagator-cv.svg propagated-cm.svg plot-weights

cl-train: classifier.dat
pr-train: propagator.dat
test: predictions-test-cm.svg propagated-test-cm.svg
truths: $(PR_TRAIN_TRUTHS)
feats: $(PR_TRAIN_DATA:%=%-truth-feats.csv.gz) $(PR_TRAIN_DATA:%=%-data-feats.csv.gz)
predictions: $(PR_TRAIN_PREDS) $(PR_TEST_PREDS)
propagated: $(PR_TRAIN_PROPS) $(PR_TEST_PROPS)
data: $(PR_TRAIN_DATAS)
weights: $(PR_TRAIN_DATA:%=%-data-dissim.csv.gz)
plot-weights: $(PR_TRAIN_DATA:%=%-data-dissim.svg)
errors: $(PR_TRAIN_DATA:%=%-data-errors.html)

clean:
	rm -f classifier.dat *-lp.dat *-chaincodes.dat dictionary.dat 2>/dev/null
	rm -f propagator.dat 2>/dev/null
	rm -f *.svg 2>/dev/null
	rm -f *.html *.txt 2>/dev/null
	rm -f *-predictions.csv.gz *-propagated.csv.gz 2>/dev/null
	rm -f *-feats.csv.gz *-truth.csv *-data.csv *-dissim.csv.gz *-distances.dat 2>/dev/null

local-images: $(PR_ALL_DATA:%=%-data.csv)
	$(LOCAL_COPY) -r ~/vt_project/Segmentation/Fine/MetaSys/ -o ../images/ $^
	ln -s ../images/ image

SpecificStages-truth.csv: truth/SpecificStages*.csv
	$(CONVERT_TRUTH) -p $(IMAGE_PREFIX) -o . 'truth/SpecificStages*.csv'
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

%-feats.csv.gz: %.csv chaincode.args
	$(FEATURES) -m chaincode -a chaincode.args -l $< -o .

classifier.dat: $(CL_TRAIN_DATA)-truth.csv $(CL_TRAIN_DATA)-truth-feats.csv.gz svm.args
	$(CL_TRAIN) -m svm -a svm.args -f $(CL_TRAIN_DATA)-truth-feats.csv.gz -t $(CL_TRAIN_DATA)-truth.csv -o .

%-predictions.csv.gz: %.csv classifier.dat
	$(CL_PREDICT) -l $< -m classifier.dat -o .

predictions-cm.svg: $(PR_TRAIN_PREDS) $(PR_TRAIN_TRUTHS)
	$(EVALUATE) --all-prefix=predictions -p $(PR_TRAIN_PREDS) -t $(PR_TRAIN_TRUTHS) -o .

predictions-test-cm.svg: $(PR_TEST_PREDS) $(PR_TEST_TRUTHS)
	$(EVALUATE) --all-prefix=predictions-test -p $(PR_TEST_PREDS) -t $(PR_TEST_TRUTHS) -o .

%-data-predictions-cm.svg: %-data-predictions.csv.gz %-truth.csv
	$(EVALUATE) -p $*-data-predictions.csv.gz -t $*-truth.csv -o .

%-data-dissim.csv.gz: %-data.csv dissim.args %-truth.csv
	$(DISSIM) -m expression -a dissim.args -l $< -o .

%-data-dissim.svg: %-data-dissim.csv.gz %-truth.csv
	$(PLOT_WEIGHTS) -w $< -o . -t $*-truth.csv

propagator.dat: $(PR_TRAIN_DISSIMS) $(PR_TRAIN_TRUTHS) $(PR_TRAIN_PREDS) harmonic.args
	$(PR_TRAIN) -m harmonic -a harmonic.args -t $(PR_TRAIN_TRUTHS) -d $(PR_TRAIN_DISSIMS) -p $(PR_TRAIN_PREDS) -o .

propagator-cv.svg: propagator.dat
	$(PLOT_CV) -m propagator.dat -o .

%-propagated.csv.gz: %-dissim.csv.gz %-predictions.csv.gz propagator.dat
	$(PR_PREDICT) -m propagator.dat -d $*-dissim.csv.gz -p $*-predictions.csv.gz -o .

propagated-cm.svg: $(PR_TRAIN_PROPS) $(PR_TRAIN_TRUTHS)
	$(EVALUATE) --all-prefix=propagated -p $(PR_TRAIN_PROPS) -t $(PR_TRAIN_TRUTHS) -o .

propagated-test-cm.svg: $(PR_TEST_PROPS) $(PR_TEST_TRUTHS)
	$(EVALUATE) --all-prefix=propagated-test -p $(PR_TEST_PROPS) -t $(PR_TEST_TRUTHS) -o .

%-data-propagated-cm.svg: %-data-propagated.csv.gz %-truth.csv
	$(EVALUATE) -p $*-data-propagated.csv.gz -t $*-truth.csv -o .

%-data-neighbors.html: %-data.csv %-data-dissim.csv.gz %-truth.csv %-data-predictions.csv.gz %-data-propagated.csv.gz
	$(VIZU_KNN) -l $*-data.csv -d $*-data-dissim.csv.gz -t $*-truth.csv -p $*-data-predictions.csv.gz -q $*-data-propagated.csv.gz -o .

%-data-errors.html: %-data.csv %-data-dissim.csv.gz %-truth.csv %-data-predictions.csv.gz %-data-propagated.csv.gz
	$(VIZU_KNN) --only-errors -l $*-data.csv -d $*-data-dissim.csv.gz -t $*-truth.csv -p $*-data-predictions.csv.gz -q $*-data-propagated.csv.gz -o .

errors-neighbors.html:
	$(VIZU_KNN) --only-errors -l $(PR_TRAIN_DATAS) -d $(PR_TRAIN_DISSIMS) -t $(PR_TRAIN_TRUTHS) -p $(PR_TRAIN_PREDS) -q $(PR_TRAIN_PROPS) -o .
