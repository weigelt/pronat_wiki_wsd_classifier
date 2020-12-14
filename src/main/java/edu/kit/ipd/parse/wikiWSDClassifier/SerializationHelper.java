package edu.kit.ipd.parse.wikiWSDClassifier;

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

	private static synchronized FSTConfiguration getFSTConfig() {
		if (SerializationHelper.conf == null) {
			SerializationHelper.conf = SerializationHelper.createConf();
		}
		return SerializationHelper.conf;
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
		SerializationHelper.write(classifier, outputFile);
	}

	public static void serializeEfficientNaiveBayesClassifierNative(EfficientNaiveBayes classifier, String outputFile) {
		try {
			SerializationHelper.writeNative(outputFile, classifier);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
	}

	public static void serializeEfficientNaiveBayesClassifierNativeZipped(EfficientNaiveBayes classifier, String outputFile) {
		try {
			SerializationHelper.writeNativeZipped(outputFile, classifier);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
	}

	public static void serializeFilter(Filter filter, String outputFile) {
		SerializationHelper.write(filter, outputFile);
	}

	public static void serializeFilterNative(Filter filter, String outputFile) {
		try {
			SerializationHelper.writeNative(outputFile, filter);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
	}

	public static void serializeFilterNativeZipped(Filter filter, String outputFile) {
		try {
			SerializationHelper.writeNativeZipped(outputFile, filter);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
	}

	public static void serializeInstances(Instances instances, String outputFile) {
		SerializationHelper.write(instances, outputFile);
	}

	public static void serializeInstancesNative(Instances instances, String outputFile) {
		try {
			SerializationHelper.writeNative(outputFile, instances);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
	}

	public static void serializeInstancesNativeZipped(Instances instances, String outputFile) {
		try {
			SerializationHelper.writeNativeZipped(outputFile, instances);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
	}

	public static Optional<EfficientNaiveBayes> deserializeEfficientNaiveBayesClassifier(InputStream input) {
		Object object = SerializationHelper.read(input);
		if (object instanceof EfficientNaiveBayes) {
			return Optional.of((EfficientNaiveBayes) object);
		}
		return Optional.empty();
	}

	public static Optional<EfficientNaiveBayes> deserializeEfficientNaiveBayesClassifierNative(InputStream input) {
		Object classifier = null;
		try {
			classifier = SerializationHelper.readNative(input);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
		if (classifier instanceof EfficientNaiveBayes) {
			return Optional.of((EfficientNaiveBayes) classifier);
		}
		return Optional.empty();
	}

	public static Optional<EfficientNaiveBayes> deserializeEfficientNaiveBayesClassifierNativeZipped(InputStream input) {
		Object classifier = null;
		try {
			classifier = SerializationHelper.readNativeZipped(input);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
		if (classifier instanceof EfficientNaiveBayes) {
			return Optional.of((EfficientNaiveBayes) classifier);
		}
		return Optional.empty();
	}

	public static Optional<Filter> deserializeFilter(InputStream input) {
		Object object = SerializationHelper.read(input);
		if (object instanceof Filter) {
			return Optional.of((Filter) object);
		}
		return Optional.empty();
	}

	public static Optional<Filter> deserializeFilterNative(InputStream input) {
		Object filter = null;
		try {
			filter = SerializationHelper.readNative(input);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
		if (filter instanceof Filter) {
			return Optional.of((Filter) filter);
		}
		return Optional.empty();
	}

	public static Optional<Filter> deserializeFilterNativeZipped(InputStream input) {
		Object filter = null;
		try {
			filter = SerializationHelper.readNativeZipped(input);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
		if (filter instanceof Filter) {
			return Optional.of((Filter) filter);
		}
		return Optional.empty();
	}

	public static Optional<Instances> deserializeInstances(InputStream input) {
		Object object = SerializationHelper.read(input);
		if (object instanceof Instances) {
			return Optional.of((Instances) object);
		}
		return Optional.empty();
	}

	public static Optional<Instances> deserializeInstancesNative(InputStream input) {
		Object instances = null;
		try {
			instances = SerializationHelper.readNative(input);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
		if (instances instanceof Instances) {
			return Optional.of((Instances) instances);
		}
		return Optional.empty();
	}

	public static Optional<Instances> deserializeInstancesNativeZipped(InputStream input) {
		Object instances = null;
		try {
			instances = SerializationHelper.readNativeZipped(input);
		} catch (Exception e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
		if (instances instanceof Instances) {
			return Optional.of((Instances) instances);
		}
		return Optional.empty();
	}

	private static void write(Object object, String outputFileName) {
		try (FileOutputStream stream = new FileOutputStream(outputFileName);
				OutputStream out = new DeflaterOutputStream(stream);
				FSTObjectOutput fstOut = SerializationHelper.getFSTConfig().getObjectOutput(out);) {
			fstOut.writeObject(object);
		} catch (IOException e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
	}

	private static void writeNative(String outputFileName, Object object) {
		try (FileOutputStream outputStream = new FileOutputStream(outputFileName);
				ObjectOutputStream objectOutputStream = new ObjectOutputStream(outputStream);) {
			objectOutputStream.writeObject(object);
		} catch (IOException e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
	}

	private static void writeNativeZipped(String outputFileName, Object object) {
		try (FileOutputStream outputStream = new FileOutputStream(outputFileName);
				GZIPOutputStream zippedStream = new GZIPOutputStream(outputStream);
				ObjectOutputStream objectOutputStream = new ObjectOutputStream(zippedStream);) {
			objectOutputStream.writeObject(object);
		} catch (IOException e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
	}

	private static Object read(InputStream input) {
		try (InputStream in = new InflaterInputStream(input);
				FSTObjectInput fstIn = SerializationHelper.getFSTConfig().getObjectInput(in);) {
			return fstIn.readObject();
		} catch (ClassNotFoundException | IOException e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
		return null;
	}

	private static Object readNative(InputStream input) {
		try (ObjectInputStream in = new ObjectInputStream(input)) {
			return in.readObject();
		} catch (ClassNotFoundException | IOException e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
		return null;
	}

	private static Object readNativeZipped(InputStream input) {
		try (GZIPInputStream zippedStream = new GZIPInputStream(input); ObjectInputStream in = new ObjectInputStream(zippedStream);) {
			return in.readObject();
		} catch (ClassNotFoundException | IOException e) {
			SerializationHelper.logger.warn(e.getMessage(), e.getCause());
		}
		return null;
	}

}
