package edu.kit.ipd.parse.wikiWSDClassifier;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Optional;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.InflaterInputStream;

import org.nustaq.serialization.FSTConfiguration;
import org.nustaq.serialization.FSTObjectInput;
import org.nustaq.serialization.FSTObjectOutput;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

public class SerializationHelper {
    private static final FSTConfiguration conf = createConf();

    public static void serializeEfficientNaiveBayesClassifier(EfficientNaiveBayes classifier, String outputFile) {
        write(classifier, outputFile);
    }

    private static FSTConfiguration createConf() {
        final FSTConfiguration conf = FSTConfiguration.createDefaultConfiguration();
        conf.registerClass(Filter.class, StringToNominal.class, Instances.class, EfficientNaiveBayes.class);
        conf.setShareReferences(false);
        return conf;
    }

    public static void serializeFilter(Filter filter, String outputFile) {
        write(filter, outputFile);
    }

    public static void serializeInstances(Instances instances, String outputFile) {
        write(instances, outputFile);
    }

    public static Optional<Instances> deserializeInstances(String inputFile) {
        Object object = read(inputFile);
        if (object instanceof Instances) {
            return Optional.of((Instances) object);
        }
        return Optional.empty();
    }

    public static Optional<EfficientNaiveBayes> deserializeEfficientNaiveBayesClassifier(String inputFile) {
        Object object = read(inputFile);
        if (object instanceof EfficientNaiveBayes) {
            return Optional.of((EfficientNaiveBayes) object);
        }
        return Optional.empty();
    }

    public static Optional<Filter> deserializeFilter(String inputFile) {
        Object object = read(inputFile);
        if (object instanceof Filter) {
            return Optional.of((Filter) object);
        }
        return Optional.empty();
    }

    private static void write(Object object, String outputFileName) {
        try {
            FileOutputStream stream = new FileOutputStream(outputFileName);
            OutputStream out = new DeflaterOutputStream(stream);
            FSTObjectOutput fstOut = conf.getObjectOutput(out);
            fstOut.writeObject(object);
            fstOut.close();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Object read(String inputFileName) {
        try {
            FileInputStream stream = new FileInputStream(inputFileName);
            InputStream in = new InflaterInputStream(stream);
            FSTObjectInput fstIn = conf.getObjectInput(in);
            Object object = fstIn.readObject();
            fstIn.close();
            in.close();
            return object;
        } catch (ClassNotFoundException | IOException e) {
            e.printStackTrace();
        }
        return null;
    }

}
