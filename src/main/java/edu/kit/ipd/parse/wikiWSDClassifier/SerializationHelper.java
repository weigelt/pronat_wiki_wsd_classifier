package edu.kit.ipd.parse.wikiWSDClassifier;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Hashtable;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.InflaterInputStream;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NominalAttributeInfo;
import weka.core.Queue;
import weka.core.Range;
import weka.core.RelationalLocator;
import weka.core.SerializedObject;
import weka.core.StringLocator;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

public class SerializationHelper {

    public static void serializeEfficientNaiveBayesClassifier(EfficientNaiveBayes classifier, String outputFile) {
        write(classifier, getKryoForEfficientNaiveBayes(), outputFile);
    }

    public static void serializeFilter(Filter filter, String outputFile) {
        if (!filter.getClass()
                   .equals(StringToNominal.class)) {
            throw new IllegalArgumentException("Currently, only StringToNominal-Filters are allowed");
        }
        write(filter, getKryoForStringToNominalFilter(), outputFile);
    }

    public static void serializeInstances(Instances instances, String outputFile) {
        write(instances, getKryoForInstances(), outputFile);
    }

    public static Optional<Instances> deserializeInstances(String inputFile) {
        Object object = read(Instances.class, getKryoForInstances(), inputFile);
        if (object instanceof Instances) {
            return Optional.of((Instances) object);
        }
        return Optional.empty();
    }

    public static Optional<EfficientNaiveBayes> deserializeEfficientNaiveBayesClassifier(String inputFile) {
        Object object = read(EfficientNaiveBayes.class, getKryoForEfficientNaiveBayes(), inputFile);
        if (object instanceof EfficientNaiveBayes) {
            return Optional.of((EfficientNaiveBayes) object);
        }
        return Optional.empty();
    }

    public static Optional<Filter> deserializeFilter(String inputFile) {
        Object object = read(Filter.class, getKryoForStringToNominalFilter(), inputFile);
        if (object instanceof Filter) {
            return Optional.of((Filter) object);
        }
        return Optional.empty();
    }

    private static void write(Object object, Kryo kryo, String outputFileName) {
        try {
            FileOutputStream fos = new FileOutputStream(outputFileName);
            OutputStream out = new DeflaterOutputStream(fos);
            Output output = new Output(out);
            kryo.writeObject(output, object);
            output.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static Object read(Class<?> cls, Kryo kryo, String inputFileName) {
        try {
            FileInputStream fis = new FileInputStream(inputFileName);
            InputStream in = new InflaterInputStream(fis);
            Input input = new Input(in);
            Object object = kryo.readObject(input, cls);
            input.close();
            return object;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    private static Kryo getKryoForEfficientNaiveBayes() {
        Kryo kryo = new Kryo();
        kryo.register(EfficientNaiveBayes.class);
        kryo.register(SparseDiscreteEstimator.class, new SparseDiscreteEstimatorSerializer());
        kryo.register(SparseDiscreteEstimator[].class);
        kryo.register(SparseDiscreteEstimator[][].class);
        kryo.register(ConcurrentHashMap.class);
        kryo.register(AtomicReference.class);
        kryo.register(ArrayList.class);
        kryo.register(Attribute.class, new AttributeSerializer());
        kryo.register(NominalAttributeInfo.class);
        kryo.register(Hashtable.class);
        kryo.register(SerializedObject.class);
        kryo.register(byte[].class);
        kryo.register(Instances.class, new InstancesSerializer());
        return kryo;
    }

    private static Kryo getKryoForStringToNominalFilter() {
        Kryo kryo = new Kryo();
        kryo.register(Filter.class);
        kryo.register(StringToNominal.class);
        kryo.register(Range.class);
        kryo.register(boolean[].class);
        kryo.register(ArrayList.class);
        kryo.register(Attribute.class, new AttributeSerializer());
        kryo.register(NominalAttributeInfo.class);
        kryo.register(Hashtable.class);
        kryo.register(SerializedObject.class);
        kryo.register(byte[].class);
        kryo.register(Instances.class, new InstancesSerializer());
        kryo.register(RelationalLocator.class);
        kryo.register(int[].class);
        kryo.register(BitSet.class);
        kryo.register(StringLocator.class);
        kryo.register(Queue.class);
        return kryo;
    }

    private static Kryo getKryoForInstances() {
        Kryo kryo = new Kryo();
        kryo.register(ArrayList.class);
        kryo.register(Attribute.class, new AttributeSerializer());
        kryo.register(NominalAttributeInfo.class);
        kryo.register(Hashtable.class);
        kryo.register(SerializedObject.class);
        kryo.register(byte[].class);
        kryo.register(Instances.class, new InstancesSerializer());
        return kryo;
    }

    private static class SparseDiscreteEstimatorSerializer extends Serializer<SparseDiscreteEstimator> {

        @Override
        public void write(Kryo kryo, Output output, SparseDiscreteEstimator object) {
            output.writeInt(object.getNumSymbols());
            output.writeDouble(object.getPrior());
            output.writeDouble(object.getSumOfCounts());
            kryo.writeObject(output, object.getCounts());
        }

        @SuppressWarnings("unchecked")
        @Override
        public SparseDiscreteEstimator read(Kryo kryo, Input input, Class<? extends SparseDiscreteEstimator> type) {
            SparseDiscreteEstimator estimator = new SparseDiscreteEstimator();
            estimator.setNumSymbols(input.readInt());
            estimator.setfPrior(input.readDouble());
            estimator.setSumOfCounts(input.readDouble());
            ConcurrentHashMap<Integer, Double> map = kryo.readObject(input, ConcurrentHashMap.class);
            estimator.setCounts(map);
            return estimator;
        }
    }

    private static class InstancesSerializer extends Serializer<Instances> {

        @Override
        public void write(Kryo kryo, Output output, Instances object) {
            // TODO
            output.writeString(object.relationName());
            int numInstances = object.numInstances();
            output.writeInt(numInstances);
            kryo.writeObject(output, getAttributes(object));
            for (int i = 0; i < numInstances; i++) {
                Instance instance = object.get(i);
                kryo.writeObject(output, instance);
            }
        }

        private ArrayList<Attribute> getAttributes(Instances instances) {
            int numAttributes = instances.numAttributes();
            ArrayList<Attribute> attributes = new ArrayList<>(numAttributes);
            for (int i = 0; i < numAttributes; i++) {
                attributes.add(instances.attribute(i));
            }
            return attributes;
        }

        @SuppressWarnings("unchecked")
        @Override
        public Instances read(Kryo kryo, Input input, Class<? extends Instances> type) {
            // TODO
            String name = input.readString();
            int capacity = input.readInt();
            ArrayList<Attribute> attInfo = kryo.readObject(input, ArrayList.class);

            Instances instances = new Instances(name, attInfo, capacity);
            for (int i = 0; i < capacity; i++) {
                Instance instance = kryo.readObject(input, Instance.class);
                instances.add(instance);
            }
            return instances;
        }
    }

    private static class AttributeSerializer extends Serializer<Attribute> {

        @Override
        public void write(Kryo kryo, Output output, Attribute object) {
            // TODO
        }

        @Override
        public Attribute read(Kryo kryo, Input input, Class<? extends Attribute> type) {
            // TODO

            // String attributeName;
            // List<String> values;
            // ProtectedProperties metadata;
            // return new Attribute(attributeName, values, metadata);
            return null;
        }

    }

}
