package edu.kit.ipd.pronat.wikiWSDClassifier;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

/**
 * Enum representing the various classifier methods like NaiveBayes.
 *
 * @author Jan Keim
 * @author Sebastian Weigelt
 *
 */
public enum ClassifierMethod {
	EfficientNaiveBayes, NaiveBayes; // J48, RandomForest;

	public Classifier getClassifier() {
		switch (this) {
		case EfficientNaiveBayes:
			return new EfficientNaiveBayes();
		case NaiveBayes:
			return new NaiveBayes();
		default:
			return new EfficientNaiveBayes();
		}
	}
}
