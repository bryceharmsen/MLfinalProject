package ml;

import java.io.File;
import java.io.FileWriter;
import java.io.FileInputStream;
import java.io.IOException;

import java.util.ArrayList;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import weka.classifiers.Evaluation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.functions.SMO;

public class App {

    private String arffDir;
    private String resultsDir;
    private int numTuningTrials;
    private int fileCounter;

    public App(String arffDir, String resultsDir, int numTuningTrials) {
        this.arffDir = arffDir;
        this.resultsDir = resultsDir;
        this.numTuningTrials = numTuningTrials;
        this.fileCounter = 0;
    }


    public void run() throws IOException, Exception {
        FilePair[] filePairs = this.getFilePairsIn(new File(this.arffDir));
        NamedClassifier[] classifiers = {
            new NamedClassifier(new MultilayerPerceptron(), "MLP"),
            new NamedClassifier(new SMO(), "SVM"),
            new NamedClassifier(new RBFNetwork(), "RBFNetwork"),
            new NamedClassifier(new RandomForest(), "RandomForest")
        };

        for (FilePair pair : filePairs) {
            Instances trainingInstances = this.getInstancesFrom(pair.getTrainingFile()),
                        testingInstances = this.getInstancesFrom(pair.getTestingFile());
            InstancesPair instancesPair = new InstancesPair(trainingInstances, testingInstances);
            this.runForInstancesPair(classifiers, instancesPair);
            this.fileCounter++;
        }
    }

    public void runForInstancesPair(NamedClassifier[] classifiers, InstancesPair instancesPair) throws IOException, Exception {
        for (NamedClassifier namedClassifier : classifiers) {
            AbstractClassifier classifier = namedClassifier.getClassifier();

            OptionsBuilder builder = this.getOptionsBuilder(namedClassifier.getName(), this.numTuningTrials); //TODO give user control over number of tuning sets
            Options[] generatedOptions = builder.getGeneratedOptions();
            System.out.println(builder.toString());
            ClassifierRunner runner = new ClassifierRunner(classifier, instancesPair, generatedOptions, classifier.toString());

            Timer t = new Timer();
            t.start();
            Evaluation eval = runner.getBestEvaluation();
            t.end();

            this.writeEvalToFile(eval, runner.getBestOptions(), t, namedClassifier.getName() + Integer.toString(this.fileCounter) + ".txt");
        }
    }

    public OptionsBuilder getOptionsBuilder(String classifierName, int numGeneratedOptions) throws Exception {
        OptionsBuilder generatedOptions;
        switch(classifierName) {
            case "MLP":
                generatedOptions = new MlpOptionsBuilder(numGeneratedOptions);
                break;
            case "SVM":
                generatedOptions = new SvmOptionsBuilder(numGeneratedOptions);
                break;
            case "RBFNetwork":
                generatedOptions = new RbfOptionsBuilder(numGeneratedOptions);
                break;
            case "RandomForest":
                generatedOptions = new RandForestOptionsBuilder(numGeneratedOptions);
                break;
            default:
                throw new Exception("Cannot build options for unknown classifier");
        }
        
        return generatedOptions;
    }

    public FilePair[] getFilePairsIn(File dir) throws IOException, Exception {
        ArrayList<FilePair> filePairs = new ArrayList<>();
        
        if (!dir.isDirectory())
            throw new IOException(dir.getName() + " is not a directory");
        
        for (File subDir : dir.listFiles()) {
            if (subDir.isDirectory())
                filePairs.add(new FilePair(subDir.listFiles()));
        }

        FilePair[] pairs = new FilePair[filePairs.size()];
        return filePairs.toArray(pairs);
    }

    public void writeEvalToFile(Evaluation eval, Options bestOptions, Timer t, String classifierName) throws IOException, Exception {
        File resultsFile = new File(this.resultsDir + classifierName);
        FileWriter writer = new FileWriter(resultsFile);
        writer.write("====  TIME  ====\n" + Double.toString(t.getTime() / Math.pow(10, 9)) + " seconds\n\n");
        writer.write("====  OPTIONS  ====\n" + bestOptions.toString() + "\n\n");
        writer.write("====  SUMMARY  ====\n" + eval.toSummaryString());
        writer.close();
    }

    public Instances getInstancesFrom(File file) throws Exception {
        DataSource source = new DataSource(new FileInputStream(file));
        return source.getDataSet();
    }

    public static void main(String[] args) {
        try {
            System.out.println("Reading command line args for <arff directory> <results directory> <number of tuning trials>");
            if (args.length != 2) {
                args = new String[] {"./arffs/", "./results/", "10"};
                System.out.println("NOTICE: Defaulting to " + args[0] + ", " + args[1] + " and " + args[2] + " for CLI args.");
            }
            App app = new App(args[0], args[1], Integer.valueOf(args[2]));
            app.run();
        } catch (IOException ioe) {
            System.out.println("IOException caught: " + ioe);
            ioe.printStackTrace();
        } catch (Exception e) {
            System.out.println("Exception cuaght: " + e);
            e.printStackTrace();
        }
    }
}