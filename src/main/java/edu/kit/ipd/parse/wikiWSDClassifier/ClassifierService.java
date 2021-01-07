package edu.kit.ipd.parse.wikiWSDClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.SortedSet;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

/**
 * @author Jan Keim
 *
 */
public class ClassifierService {
	private static final Logger logger = LoggerFactory.getLogger(ClassifierService.class);

	private Classifier classifier;
	private Filter filter;
	private Instances header;

	// Filterlist for stuff, that might occur and we don't want
	public static List<String> filterWords = Arrays.asList("NONE", ".", ",", ";", "-rrb-", "-rsb-", "-lrb-", "-lsb-", "\'\'", "\'", "--",
			"-", ":", "``", "`", "|", "!", "?", "<", ">", "_", "−", "#", "...", "-lcb-", "-rcb-", "<math>", "\\", "</sup>", "<sup>", "+",
			"ii", "iii", "</u>", "<u>", "</tt>", "<tt>", "=");
	public static List<String> additionalFilterWords = Arrays.asList("'s", "%");

	public ClassifierService(Classifier classifier, Filter filter) {
		this.classifier = classifier;
		this.filter = filter;
		header = null;
	}

	public ClassifierService(Classifier classifier, Filter filter, Instances header) {
		this.classifier = classifier;
		this.filter = filter;
		this.header = header;
		this.header.setAttributeWeight(1, 10.);
	}

	/**
	 * Classifies the instance and return the value of the classified instance. The
	 * class attribute has to be a nominal or string attribute, otherwise an empty
	 * string.
	 *
	 * @param instance
	 * @return
	 */
	public Classification classifyInstance(Instance instance) {
		if ((classifier == null) || (filter == null)) {
			throw new IllegalStateException("Classifier or Filter are null!");
		}
		Classification c = Classification.empty();
		Instance instanceCopy = new DenseInstance(instance);
		instanceCopy.setDataset(instance.dataset());
		if (!instanceIsFiltered(instanceCopy)) {
			// attributes are not nominal, they need to be filtered first!
			try {
				filter.input(instanceCopy);
			} catch (Exception e) {
				logger.warn(e.getMessage(), e.getCause());
				return c;
			}
			instanceCopy = filter.output();
		}
		try {
			instanceCopy.attribute(1).setWeight(10.);
			double classification = classifier.classifyInstance(instanceCopy);
			c = new Classification(instance.classAttribute().value((int) classification));
		} catch (Exception e) {
			logger.warn(e.getMessage(), e.getCause());
		}
		return c;
	}

	protected double[] getDistributionArray(Instance instance) throws Exception {
		if (classifier instanceof EfficientNaiveBayes) {
			return ((EfficientNaiveBayes) classifier).logDistributionForInstance(instance);
		} else {
			logger.warn("Not using the EfficientNaiveBayes, thus not using thee more precise logarithmic distribution.");
			return classifier.distributionForInstance(instance);
		}
	}

	/**
	 * Always paired entries: First the Classification, then followed by its
	 * probability.
	 *
	 * @param instance
	 * @return
	 */
	public Classification[] classifyInstanceTop3(Instance instance) {
		if ((classifier == null) || (filter == null)) {
			throw new IllegalStateException("Classifier or Filter are null!");
		}
		Classification[] retArray = emptyTop3Classification();
		Instance instanceCopy = new DenseInstance(instance);
		instanceCopy.setDataset(instance.dataset());
		if (!instanceIsFiltered(instanceCopy)) {
			// attributes are not nominal, they need to be filtered first!
			try {
				filter.input(instanceCopy);
			} catch (Exception e) {
				logger.warn(e.getMessage(), e.getCause());
				return retArray;
			}
			instanceCopy = filter.output();
		}
		double[] distributionArray = new double[0];
		try {
			instanceCopy.attribute(1).setWeight(10.);
			distributionArray = getDistributionArray(instanceCopy);
		} catch (Exception e) {
			logger.warn(e.getMessage(), e.getCause());
			return emptyTop3Classification();
		}
		// get the top 3
		for (int i = 0; i < distributionArray.length; i++) {
			Classification c = new Classification(instance.classAttribute().value(i), distributionArray[i]);
			if (c.compareTo(retArray[0]) > 0) {
				retArray[2] = retArray[1];
				retArray[1] = retArray[0];
				retArray[0] = c;
			} else if (c.compareTo(retArray[1]) > 0) {
				retArray[2] = retArray[1];
				retArray[1] = c;
			} else if (c.compareTo(retArray[2]) > 0) {
				retArray[2] = c;
			}
		}
		return retArray;
	}

	public Classification classifyInstanceWithLemma(Instance instance, String lemma) {
		// disambiguate
		Classification clazz;
		Classification[] top3clazz = classifyInstanceTop3(instance);
		clazz = top3clazz[0];
		for (int i = 2; i >= 0; i--) {
			if (top3clazz[i].getClassificationString().toLowerCase().contains(lemma) && distributionIsSimilar(top3clazz[0], top3clazz[i])) {
				clazz = top3clazz[i];
			}
		}
		if (logger.isDebugEnabled()) {
			String disamStr = String.format("%s -> %s -- %s; %s; %s", lemma, clazz.getClassificationString(), top3clazz[0], top3clazz[1],
					top3clazz[2]);
			logger.debug(disamStr);
		}
		return clazz;
	}

	private boolean distributionIsSimilar(Classification main, Classification other) {
		double threshold = 0.75;
		double relation = 0;
		if (other.getProbability() < 0) {
			relation = main.getProbability() / other.getProbability();
			return relation >= threshold;
		} else {
			relation = main.getProbability() - other.getProbability();
			return relation <= (1 - threshold);
		}
	}

	private boolean instanceIsFiltered(Instance instance) {
		for (int i = 0; i < instance.numAttributes(); i++) {
			if (!instance.attribute(i).isNominal()) {
				return false;
			}
		}
		return true;
	}

	private Classification[] emptyTop3Classification() {
		return new Classification[] { Classification.empty(), Classification.empty(), Classification.empty() };
	}

	/**
	 * @return the header as Optional
	 */
	public Optional<Instances> getHeader() {
		return Optional.ofNullable(header);
	}

	/**
	 * The standard Attributes used for classification (WSD)
	 *
	 * @return ArrayList with attributes
	 */
	public static ArrayList<Attribute> getAttributes() {
		ArrayList<Attribute> attributes = new ArrayList<>();

		// Declare the class attribute (as string attribute)
		Attribute wordSenseAttribute = new Attribute("wordSense", true);
		attributes.add(wordSenseAttribute);

		// Declare the feature vector
		Attribute actualWordAttribute = new Attribute("actualWord", true);
		actualWordAttribute.setWeight(10d);
		attributes.add(actualWordAttribute);
		attributes.add(new Attribute("actualWordPOS", true));
		attributes.add(new Attribute("word-3", true));
		attributes.add(new Attribute("word-3POS", true));
		attributes.add(new Attribute("word-2", true));
		attributes.add(new Attribute("word-2POS", true));
		attributes.add(new Attribute("word-1", true));
		attributes.add(new Attribute("word-1POS", true));
		attributes.add(new Attribute("word+1", true));
		attributes.add(new Attribute("word+1POS", true));
		attributes.add(new Attribute("word+2", true));
		attributes.add(new Attribute("word+2POS", true));
		attributes.add(new Attribute("word+3", true));
		attributes.add(new Attribute("word+3POS", true));
		attributes.add(new Attribute("leftNN", true));
		attributes.add(new Attribute("leftVB", true));
		attributes.add(new Attribute("rightNN", true));
		attributes.add(new Attribute("rightVB", true));

		return attributes;
	}

	/**
	 * Creates the header for instances with name, attributes and set class index
	 *
	 * @return Instances-header with name, attributes and set class index
	 */
	public static Instances getEmptyInstancesHeader() {
		ArrayList<Attribute> attributes = ClassifierService.getAttributes();
		// Create training set with attributes from above and set the class
		// index
		Instances header = new Instances("WordSenseDisambiguation", attributes, 0);
		header.setClassIndex(0);
		header.setAttributeWeight(1, 10d);
		return header;
	}

	/**
	 * Creates the header for instances with name, attributes and set class index.
	 * Uses the provided attributes to set to the header.
	 *
	 * @param attributes
	 *            Attributes that should be set to the header
	 * @return Instances-header with name, attributes and set class index
	 */
	public static Instances getEmptyInstancesHeader(ArrayList<Attribute> attributes) {
		Instances header = new Instances("WordSenseDisambiguation", attributes, 0);
		header.setClassIndex(0);
		header.setAttributeWeight(1, 10d);
		return header;
	}

	/**
	 * Returns the value of the class attribute of the given instance.
	 *
	 * @param instance
	 *            Instance that should be handled
	 * @return Value of the class attribute
	 */
	protected Optional<String> getStringOfClassAttributeValue(Instance instance) {
		return Optional.ofNullable(instance.stringValue(instance.classAttribute()));
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		String classifierString = classifier != null ? classifier.getClass().getName() : "null";
		String filterString = classifier != null ? filter.getClass().getName() : "null";
		return "ClassifierService [classifier=" + classifierString + ", filter=" + filterString + "]";
	}

	/**
	 * Same as {@link #classifyInstanceTop3(Instance)}. But provide top x
	 * classifications.
	 *
	 * @param instance
	 *            the instance
	 * @param lemma
	 *            the lemma
	 * @param maxHypothesis
	 *            the max amount of classifications
	 * @return a sorted list (max first) list of classifications
	 * @author Dominik Fuchss
	 */
	public List<Classification> classifyInstanceWithLemma(Instance instance, String lemma, int maxHypothesis) {
		if ((classifier == null) || (filter == null)) {
			throw new IllegalStateException("Classifier or Filter are null!");
		}

		Instance instanceCopy = new DenseInstance(instance);
		instanceCopy.setDataset(instance.dataset());
		if (!instanceIsFiltered(instanceCopy)) {
			// attributes are not nominal, they need to be filtered first!
			try {
          filter.input(instanceCopy);
			} catch (Exception e) {
				ClassifierService.logger.warn(e.getMessage(), e.getCause());
				return List.of(Classification.empty());
			}
			instanceCopy = filter.output();
		}
		double[] distributionArray = new double[0];
		try {
			instanceCopy.attribute(1).setWeight(10.);
			distributionArray = getDistributionArray(instanceCopy);
		} catch (Exception e) {
			ClassifierService.logger.warn(e.getMessage(), e.getCause());
			return List.of(Classification.empty());
		}

		// Sort upside down (max first)
		SortedSet<Classification> classifications = new TreeSet<>((a, b) -> -a.compareTo(b));
		// get the top 3
		for (int i = 0; i < distributionArray.length; i++) {
			Classification c = new Classification(instance.classAttribute().value(i), distributionArray[i]);
			classifications.add(c);
		}

		List<Classification> result = new ArrayList<>();
		Iterator<Classification> iter = classifications.iterator();
		while (iter.hasNext() && result.size() < maxHypothesis) {
			result.add(iter.next());
		}
		return result;
	}
}
