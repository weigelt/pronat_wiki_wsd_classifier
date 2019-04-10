/**
 *
 */
package im.janke.wsdClassifier;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.*;
import weka.estimators.Estimator;

/**
 * Class that represents a slightly adapted NaiveBayes. The original was the
 * {@link NaiveBayes} implementation of Weka (that this class extends).
 *
 * If you use FrequentWords in your attribute, the attribute name must start
 * with "mFrequent" and the frequent word attributes must be the last attributes
 * in the feature vector.
 *
 * @author Jan Keim
 *
 */
public class EfficientNaiveBayes extends NaiveBayes {

	/** for serialization */
	private static final long serialVersionUID = -9126601110930963510L;

	EfficientNaiveBayes() {
		super();
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances
	 *            set of instances serving as training data
	 * @exception Exception
	 *                if the classifier has not been generated successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		// instances = new Instances(instances); //don't copy
		instances.deleteWithMissingClass();

		m_NumClasses = instances.numClasses();

		// Copy the instances >NOT<
		// m_Instances = new Instances(instances);
		m_Instances = instances;

		// Reserve space for the distributions
		m_Distributions = new Estimator[m_Instances.numAttributes() - 1][m_NumClasses];
		m_ClassDistribution = new SparseDiscreteEstimator(m_NumClasses, true);

		int attIndex = 0;
		Enumeration<Attribute> enu = m_Instances.enumerateAttributes();
		while (enu.hasMoreElements()) {
			Attribute attribute = enu.nextElement();
			for (int j = 0; j < m_Instances.numClasses(); j++) {
				switch (attribute.type()) {
				case Attribute.NOMINAL:
					m_Distributions[attIndex][j] = new SparseDiscreteEstimator(attribute.numValues(), true);
					break;
				default:
					throw new Exception("Attribute type unknown to MyNaiveBayes");
				}
			}
			attIndex++;
		}

		// Compute counts
		ExecutorService executor = Executors.newWorkStealingPool();
		for (Instance instance : m_Instances) {
			executor.execute(() -> {
				try {
					updateClassifier(instance);
				} catch (Exception e) {
					e.printStackTrace();
				}
			});
		}
		executor.shutdown();
		executor.awaitTermination(42, TimeUnit.MINUTES);
		executor = null;
		// Save space
		m_Instances = new Instances(m_Instances, 0);
	}

	/**
	 * Updates the classifier with the given instance.
	 *
	 * @param instance
	 *            the new training instance to include in the model
	 * @exception Exception
	 *                if the instance could not be incorporated in the model.
	 */
	@Override
	public void updateClassifier(Instance instance) throws Exception {
		if (!instance.classIsMissing()) {
			Enumeration<Attribute> enumAtts = m_Instances.enumerateAttributes();
			int attIndex = 0;
			while (enumAtts.hasMoreElements()) {
				Attribute attribute = enumAtts.nextElement();
				if (!instance.isMissing(attribute)) {
					synchronized (m_Distributions[attIndex][(int) instance.classValue()]) {
						m_Distributions[attIndex][(int) instance.classValue()].addValue(instance.value(attribute), instance.weight());
					}
				}
				attIndex++;
			}
			synchronized (m_ClassDistribution) {
				m_ClassDistribution.addValue(instance.classValue(), instance.weight());
			}
		}
	}

	public EfficientNaiveBayes aggregate(EfficientNaiveBayes toAggregate) throws Exception {

		// Highly unlikely that discretization intervals will match between the
		// two classifiers
		if (m_UseDiscretization || toAggregate.getUseSupervisedDiscretization()) {
			throw new Exception("Unable to aggregate when supervised discretization " + "has been turned on");
		}

		if (!m_Instances.equalHeaders(toAggregate.m_Instances)) {
			throw new Exception("Can't aggregate - data headers don't match: " + m_Instances.equalHeadersMsg(toAggregate.m_Instances));
		}

		((Aggregateable) m_ClassDistribution).aggregate(toAggregate.m_ClassDistribution);

		// aggregate all conditional estimators
		for (int i = 0; i < m_Distributions.length; i++) {
			for (int j = 0; j < m_Distributions[i].length; j++) {
				((Aggregateable) m_Distributions[i][j]).aggregate(toAggregate.m_Distributions[i][j]);
			}
		}

		return this;
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see weka.classifiers.AbstractClassifier#classifyInstance(weka.core.Instance)
	 */
	// changed to work with the log-distribution stuff as the results of log(p)
	// will be negative
	@Override
	public double classifyInstance(Instance instance) throws Exception {

		double[] dist = logDistributionForInstance(instance);
		if (dist == null) {
			throw new Exception("Null distribution predicted");
		}
		switch (instance.classAttribute().type()) {
		case Attribute.NOMINAL:
			double max = Integer.MIN_VALUE; // changed
			int maxIndex = -1;

			for (int i = 0; i < dist.length; i++) {
				if (dist[i] > max) {
					maxIndex = i;
					max = dist[i];
				}
			}
			if (maxIndex >= 0) { // don't check if max>0, index instead
				return maxIndex;
			} else {
				return Utils.missingValue();
			}

		case Attribute.NUMERIC:
		case Attribute.DATE:
			return dist[0];
		default:
			return Utils.missingValue();
		}
	}

	private static final double actualWordExtraWeight = 10;

	double[] logDistributionForInstance(Instance instance) {
		if (m_UseDiscretization) {
			m_Disc.input(instance);
			instance = m_Disc.output();
		}

		// adapted with log-sum-exp trick against underflows:
		// https://stats.stackexchange.com/a/253319

		// numerator p(x|Y=C)P(Y=C)
		// with log: log(p(C_k)) + Sum(log(p(x|C_k)))
		double[] logNumerator = new double[m_NumClasses];

		for (int k = 0; k < m_NumClasses; k++) {
			// get the log(p(C_k))
			logNumerator[k] = Math.log(m_ClassDistribution.getProbability(k));
			// calc the Sum(log(p(x|C_k))) for each k
			Enumeration<Attribute> enumAtts = instance.enumerateAttributes();
			int attIndex = -1;
			while (enumAtts.hasMoreElements()) {
				attIndex++;
				Attribute a = enumAtts.nextElement();
				if (instance.isMissing(a)) {
					continue;
				}

				double probXInCk = m_Distributions[attIndex][k].getProbability(instance.value(a));
				if (attIndex == 0) {
					logNumerator[k] += actualWordExtraWeight * instance.weight() * Math.log(probXInCk);
				} else {
					logNumerator[k] += instance.weight() * Math.log(probXInCk);
				}

			}
		}

		// NOTE: to save calculation time omit the denominator.
		// this way we don't get the probabilities, but the order is still the
		// same, as we only need
		// the argmax
		return logNumerator;
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return predicted class probability distribution
	 * @exception Exception
	 *                if there is a problem generating the prediction
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] probs = new double[m_NumClasses];
		double[] logNumerator = logDistributionForInstance(instance);
		double max = Arrays.stream(logNumerator).max().orElse(Integer.MIN_VALUE);
		if (max == Integer.MAX_VALUE) {
			throw new ArithmeticException("Had a problem getting the maximum for log of numerator");
		}
		// denominator
		// probs[k] = logNumerator[k]-log(Sum(p(x|Y=C_k)p(Y=C_k)))
		// log-sum-exp trick:
		// log(Sum(a)) = log(Sum(exp(log(a)))) and
		// log(Sum(exp(a))) = A + log(Sum(exp(a-A)))
		double sum = 0;
		for (int k = 0; k < m_NumClasses; k++) {
			sum += Math.exp(logNumerator[k] - max);
		}
		double logDenominator = max + Math.log(sum);
		for (int k = 0; k < m_NumClasses; k++) {
			probs[k] = Math.exp(logNumerator[k] - logDenominator);
		}
		return probs;
	}
}