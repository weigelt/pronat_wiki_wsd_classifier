package edu.kit.ipd.parse.wikiWSDClassifier;

public class Classification implements Comparable<Classification> {
    private String classification;
    private double probability;

    Classification(String classification, double probability) {
        this.classification = classification;
        this.probability = probability;
    }

    Classification(String classification) {
        this.classification = classification;
        probability = Integer.MIN_VALUE;
    }

    @Override
    public String toString() {
        return classification + "(" + probability + ")";
    }

    /**
     * @return the classification
     */
    public String getClassificationString() {
        return classification;
    }

    /**
     * @return the probability
     */
    public double getProbability() {
        return probability;
    }

    /*
     * (non-Javadoc)
     *
     * @see java.lang.Object#hashCode()
     */
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = (prime * result) + ((classification == null) ? 0 : classification.hashCode());
        long temp;
        temp = Double.doubleToLongBits(probability);
        result = (prime * result) + (int) (temp ^ (temp >>> 32));
        return result;
    }

    /*
     * (non-Javadoc)
     *
     * @see java.lang.Object#equals(java.lang.Object)
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (this.getClass() != obj.getClass()) {
            return false;
        }
        Classification other = (Classification) obj;
        if (classification == null) {
            if (other.classification != null) {
                return false;
            }
        } else if (!classification.equals(other.classification)) {
            return false;
        }
        if (Double.doubleToLongBits(probability) != Double.doubleToLongBits(other.probability)) {
            return false;
        }
        return true;
    }

    static Classification empty() {
        return new Classification("NONE");
    }

    @Override
    public int compareTo(Classification o) {
        return Double.compare(probability, o.getProbability());
    }

}
