package edu.kit.ipd.parse.wikiWSDClassifier;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.Optional;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import java.util.zip.InflaterInputStream;

import org.nustaq.serialization.FSTConfiguration;
import org.nustaq.serialization.FSTObjectInput;
import org.nustaq.serialization.FSTObjectOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

public class SerializationHelper {
    private static final Logger logger = LoggerFactory.getLogger(SerializationHelper.class);
    private static FSTConfiguration conf;

    private synchronized static FSTConfiguration getFSTConfig() {
        if (conf == null) {
            conf = createConf();
        }
        return conf;
    }

    private SerializationHelper() {
    }

    private static FSTConfiguration createConf() {
        final FSTConfiguration conf = FSTConfiguration.createDefaultConfiguration();
        conf.registerClass(Filter.class, StringToNominal.class, Instances.class, EfficientNaiveBayes.class);
        conf.setShareReferences(false);
        return conf;
    }

    public static void serializeEfficientNaiveBayesClassifier(EfficientNaiveBayes classifier, String outputFile) {
        write(classifier, outputFile);
    }

    public static void serializeEfficientNaiveBayesClassifierNative(EfficientNaiveBayes classifier, String outputFile) {
        try {
            writeNative(outputFile, classifier);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
    }

    public static void serializeEfficientNaiveBayesClassifierNativeZipped(EfficientNaiveBayes classifier,
            String outputFile) {
        try {
            writeNativeZipped(outputFile, classifier);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
    }

    public static void serializeFilter(Filter filter, String outputFile) {
        write(filter, outputFile);
    }

    public static void serializeFilterNative(Filter filter, String outputFile) {
        try {
            writeNative(outputFile, filter);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
    }

    public static void serializeFilterNativeZipped(Filter filter, String outputFile) {
        try {
            writeNativeZipped(outputFile, filter);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
    }

    public static void serializeInstances(Instances instances, String outputFile) {
        write(instances, outputFile);
    }

    public static void serializeInstancesNative(Instances instances, String outputFile) {
        try {
            writeNative(outputFile, instances);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
    }

    public static void serializeInstancesNativeZipped(Instances instances, String outputFile) {
        try {
            writeNativeZipped(outputFile, instances);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
    }

    public static Optional<EfficientNaiveBayes> deserializeEfficientNaiveBayesClassifier(String inputFile) {
        Object object = read(inputFile);
        if (object instanceof EfficientNaiveBayes) {
            return Optional.of((EfficientNaiveBayes) object);
        }
        return Optional.empty();
    }

    public static Optional<EfficientNaiveBayes> deserializeEfficientNaiveBayesClassifierNative(String inputFile) {
        Object classifier = null;
        try {
            classifier = readNative(inputFile);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
        if ((classifier != null) && (classifier instanceof EfficientNaiveBayes)) {
            return Optional.of((EfficientNaiveBayes) classifier);
        }
        return Optional.empty();
    }

    public static Optional<EfficientNaiveBayes> deserializeEfficientNaiveBayesClassifierNativeZipped(String inputFile) {
        Object classifier = null;
        try {
            classifier = readNativeZipped(inputFile);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
        if ((classifier != null) && (classifier instanceof EfficientNaiveBayes)) {
            return Optional.of((EfficientNaiveBayes) classifier);
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

    public static Optional<Filter> deserializeFilterNative(String inputFile) {
        Object filter = null;
        try {
            filter = readNative(inputFile);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
        if ((filter != null) && (filter instanceof Filter)) {
            return Optional.of((Filter) filter);
        }
        return Optional.empty();
    }

    public static Optional<Filter> deserializeFilterNativeZipped(String inputFile) {
        Object filter = null;
        try {
            filter = readNativeZipped(inputFile);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
        if ((filter != null) && (filter instanceof Filter)) {
            return Optional.of((Filter) filter);
        }
        return Optional.empty();
    }

    public static Optional<Instances> deserializeInstances(String inputFile) {
        Object object = read(inputFile);
        if (object instanceof Instances) {
            return Optional.of((Instances) object);
        }
        return Optional.empty();
    }

    public static Optional<Instances> deserializeInstancesNative(String inputFile) {
        Object instances = null;
        try {
            instances = readNative(inputFile);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
        if ((instances != null) && (instances instanceof Instances)) {
            return Optional.of((Instances) instances);
        }
        return Optional.empty();
    }

    public static Optional<Instances> deserializeInstancesNativeZipped(String inputFile) {
        Object instances = null;
        try {
            instances = readNativeZipped(inputFile);
        } catch (Exception e) {
            logger.warn(e.getMessage(), e.getCause());
        }
        if ((instances != null) && (instances instanceof Instances)) {
            return Optional.of((Instances) instances);
        }
        return Optional.empty();
    }

    private static void write(Object object, String outputFileName) {
        try (FileOutputStream stream = new FileOutputStream(outputFileName);
                OutputStream out = new DeflaterOutputStream(stream);
                FSTObjectOutput fstOut = getFSTConfig().getObjectOutput(out);) {
            fstOut.writeObject(object);
        } catch (IOException e) {
            logger.warn(e.getMessage(), e.getCause());
        }
    }

    private static void writeNative(String outputFileName, Object object) {
        try (FileOutputStream outputStream = new FileOutputStream(outputFileName);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(outputStream);) {
            objectOutputStream.writeObject(object);
        } catch (IOException e) {
            logger.warn(e.getMessage(), e.getCause());
        }
    }

    private static void writeNativeZipped(String outputFileName, Object object) {
        try (FileOutputStream outputStream = new FileOutputStream(outputFileName);
                GZIPOutputStream zippedStream = new GZIPOutputStream(outputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(zippedStream);) {
            objectOutputStream.writeObject(object);
        } catch (IOException e) {
            logger.warn(e.getMessage(), e.getCause());
        }
    }

    private static Object read(String inputFileName) {
        try (FileInputStream stream = new FileInputStream(inputFileName);
                InputStream in = new InflaterInputStream(stream);
                FSTObjectInput fstIn = getFSTConfig().getObjectInput(in);) {
            Object object = fstIn.readObject();
            return object;
        } catch (ClassNotFoundException | IOException e) {
            logger.warn(e.getMessage(), e.getCause());
        }
        return null;
    }

    private static Object readNative(String inputFileName) {
        try (FileInputStream stream = new FileInputStream(inputFileName);
                ObjectInputStream in = new ObjectInputStream(stream);) {
            Object object = in.readObject();
            return object;
        } catch (ClassNotFoundException | IOException e) {
            logger.warn(e.getMessage(), e.getCause());
        }
        return null;
    }

    private static Object readNativeZipped(String inputFileName) {
        try (FileInputStream stream = new FileInputStream(inputFileName);
                GZIPInputStream zippedStream = new GZIPInputStream(stream);
                ObjectInputStream in = new ObjectInputStream(zippedStream);) {
            Object object = in.readObject();
            return object;
        } catch (ClassNotFoundException | IOException e) {
            logger.warn(e.getMessage(), e.getCause());
        }
        return null;
    }

}
