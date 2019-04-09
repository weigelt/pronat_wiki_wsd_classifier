/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    DiscreteEstimator.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *     Modified by Jan Keim, Karlsruhe Institute of Technology
 */

package edu.kit.ipd.parse.wikiWSDClassifier;

import java.util.HashMap;

import weka.core.Aggregateable;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.RevisionUtils;
import weka.estimators.DiscreteEstimator;
import weka.estimators.Estimator;
import weka.estimators.IncrementalEstimator;

/**
 * This is basically {@link DiscreteEstimator}, but changed for own needs.
 * Simple symbolic probability estimator based on symbol counts.
 *
 * Changed stuff by Jan Keim: use float instead of double for less memory usage
 *
 * @author Len Trigg (trigg@cs.waikato.ac.nz), Jan Keim
 * @version $Revision: 11247 $
 */
public class SparseDiscreteEstimator extends Estimator implements IncrementalEstimator, Aggregateable<SparseDiscreteEstimator> {

	/** for serialization */
	private static final long serialVersionUID = -5526486742612434779L;

	/** Hold the counts */
	private final HashMap<Integer, Double> m_Counts;
	private int numSymbols;

	/** Hold the sum of counts */
	private double m_SumOfCounts;

	/** Initialization for counts */
	private double m_FPrior = 0d;

	/**
	 * Constructor
	 *
	 * @param numSymbols
	 *            the number of possible symbols (remember to include 0)
	 * @param laplace
	 *            if true, counts will be initialized to 1
	 */
	SparseDiscreteEstimator(int numSymbols, boolean laplace) {
		m_Counts = new HashMap<>();
		this.numSymbols = numSymbols;
		m_SumOfCounts = 0;
		if (laplace) {
			m_FPrior = 1;
			m_SumOfCounts = numSymbols;
		}
	}

	/**
	 * Constructor
	 *
	 * @param nSymbols
	 *            the number of possible symbols (remember to include 0)
	 * @param fPrior
	 *            value with which counts will be initialized
	 */
	public SparseDiscreteEstimator(int nSymbols, float fPrior) {
		m_Counts = new HashMap<>();
		m_FPrior = fPrior;
		m_SumOfCounts = fPrior * nSymbols;
	}

	/**
	 * Add a new data value to the current estimator.
	 *
	 * @param data
	 *            the new data value
	 * @param weight
	 *            the weight assigned to the data value
	 */
	@Override
	public void addValue(double data, double weight) {
		synchronized (this) {
			Double old = m_Counts.get((int) data);
			if (old == null) {
				old = m_FPrior;
			}
			m_Counts.put((int) data, old + weight);
			m_SumOfCounts += weight;
		}
	}

	/**
	 * Get a probability estimate for a value
	 *
	 * @param data
	 *            the value to estimate the probability of
	 * @return the estimated probability of the supplied value
	 */
	@Override
	public double getProbability(double data) {
		synchronized (this) {
			if (m_SumOfCounts == 0) {
				return 0;
			}
			Double val = m_Counts.get((int) data);
			if (val == null) {
				val = m_FPrior;
			}
			return val / m_SumOfCounts;
		}
	}

	/**
	 * Gets the number of symbols this estimator operates with
	 *
	 * @return the number of estimator symbols
	 */
	private int getNumSymbols() {
		return numSymbols;
	}

	/**
	 * Get the count for a value
	 *
	 * @param data
	 *            the value to get the count of
	 * @return the count of the supplied value
	 */
	public double getCount(double data) {
		synchronized (this) {
			if (m_SumOfCounts == 0) {
				return 0;
			}
			Double val = m_Counts.get((int) data);
			if (val == null) {
				val = m_FPrior;
			}
			return val;
		}
	}

	/**
	 * Get the sum of all the counts
	 *
	 * @return the total sum of counts
	 */
	private double getSumOfCounts() {
		return m_SumOfCounts;
	}

	/**
	 * Display a representation of this estimator
	 */
	@Override
	public String toString() {
		StringBuffer result = new StringBuffer("Discrete Estimator.");
		result.append("  Total = ").append(m_SumOfCounts).append("\n");
		return result.toString();
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// class
		if (!m_noClass) {
			result.enable(Capability.NOMINAL_CLASS);
			result.enable(Capability.MISSING_CLASS_VALUES);
		} else {
			result.enable(Capability.NO_CLASS);
		}

		// attributes
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		return result;
	}

	/**
	 * Returns the revision string.
	 *
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 11247 $");
	}

	@Override
	public SparseDiscreteEstimator aggregate(SparseDiscreteEstimator toAggregate) throws Exception {
		synchronized (this) {
			if (toAggregate.getNumSymbols() != numSymbols) {
				throw new Exception("DiscreteEstimator to aggregate has a different " + "number of symbols");
			}

			m_SumOfCounts += toAggregate.getSumOfCounts();
			for (Integer i : toAggregate.m_Counts.keySet()) {
				Double otherVal = toAggregate.m_Counts.get(i);
				Double thisVal = m_Counts.get(i);
				if (thisVal == null) {
					thisVal = m_FPrior;
				}
				m_Counts.put(i, (thisVal + otherVal) - toAggregate.m_FPrior);
			}
			m_SumOfCounts -= (toAggregate.m_FPrior * numSymbols);
			return this;
		}
	}

	@Override
	public void finalizeAggregation() throws Exception {
		// nothing to do
	}
}
