package edu.kit.ipd.parse.wikiWSDClassifier;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Jan Keim
 *
 */
public class InstanceBuilder {
    private Instance instance;
    private static final String NONE_VAL = "NONE";

    public InstanceBuilder(Instances header) {
        instance = new DenseInstance(header.numAttributes());
        instance.setDataset(header);
        instance.setWeight(2);
    }

    public Instance build() {
        return instance;
    }

    public InstanceBuilder setActualWordWithPOS(String lemma, String pos) {
        instance.setValue(1, lemma);
        instance.setValue(2, pos);
        return this;
    }

    public InstanceBuilder set3rdLeftWithPOS(String lemma, String pos) {
        if (!lemma.equals(NONE_VAL)) {
            instance.setValue(3, lemma);
            instance.setValue(4, pos);
        }
        return this;
    }

    public InstanceBuilder set2ndLeftWithPOS(String lemma, String pos) {
        if (!lemma.equals(NONE_VAL)) {
            instance.setValue(5, lemma);
            instance.setValue(6, pos);
        }
        return this;
    }

    public InstanceBuilder set1stLeftWithPOS(String lemma, String pos) {
        if (!lemma.equals(NONE_VAL)) {
            instance.setValue(7, lemma);
            instance.setValue(8, pos);
        }
        return this;
    }

    public InstanceBuilder set1stRightWithPOS(String lemma, String pos) {
        if (!lemma.equals(NONE_VAL)) {
            instance.setValue(9, lemma);
            instance.setValue(10, pos);
        }
        return this;
    }

    public InstanceBuilder set2ndRightWithPOS(String lemma, String pos) {
        if (!lemma.equals(NONE_VAL)) {
            instance.setValue(11, lemma);
            instance.setValue(12, pos);
        }
        return this;
    }

    public InstanceBuilder set3rdRightWithPOS(String lemma, String pos) {
        if (!lemma.equals(NONE_VAL)) {
            instance.setValue(13, lemma);
            instance.setValue(14, pos);
        }
        return this;
    }

    public InstanceBuilder setLeftNoun(String lemma) {
        addAttributeToInstance(instance, 15, lemma);
        return this;
    }

    public InstanceBuilder setLeftVerb(String lemma) {
        addAttributeToInstance(instance, 16, lemma);
        return this;
    }

    public InstanceBuilder setRightNoun(String lemma) {
        addAttributeToInstance(instance, 17, lemma);
        return this;
    }

    public InstanceBuilder setRightVerb(String lemma) {
        addAttributeToInstance(instance, 18, lemma);
        return this;
    }

    private void addAttributeToInstance(Instance instance, int attrIndex, String attrValue) {
        if (!attrValue.equals(NONE_VAL)) {
            instance.setValue(attrIndex, attrValue);
        }
    }
}
