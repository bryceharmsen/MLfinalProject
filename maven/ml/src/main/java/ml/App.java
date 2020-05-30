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
    private int fileCounter;

    public App(String arffDir, String resultsDir) {
        this.arffDir = arffDir;
        this.resultsDir = resultsDir;
        this.fileCounter = 0;
    }


    public void run() {
        try {
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
            }
        } catch (IOException ioe) {
            System.out.println("IOException caught: " + ioe);
            ioe.printStackTrace();
        } catch (Exception e) {
            System.out.println("Exception cuaght: " + e);
            e.printStackTrace();
        }
    }

    public void runForInstancesPair(NamedClassifier[] classifiers, InstancesPair instancesPair) throws IOException, Exception {
        for (NamedClassifier namedClassifier : classifiers) {
            AbstractClassifier classifier = namedClassifier.getClassifier();
            
            //Enumeration<Option> enumer = cls.listOptions();
            //Iterator<Option> enumerItr = enumer.asIterator();
            //while(enumerItr.hasNext()) {
            //    Option option = enumerItr.next();
            //    System.out.println("Description: " + option.description());
            //    System.out.println("NumArgs: " + option.numArguments());
            //}
            //TODO finish OptionsBuilder

            OptionBuilder builder = new OptionBuilder(classifier.getOptions());
            Options[] optionsList = {builder.getOptions()};
            ClassifierRunner runner = new ClassifierRunner(classifier, instancesPair, optionsList, classifier.toString());

            Timer t = new Timer();
            t.start();
            Evaluation eval = runner.getBestEvaluation();
            t.end();

            this.writeEvalToFile(eval, t, namedClassifier.getName() + Integer.toString(this.fileCounter++) + ".txt");
        }
    }

    public FilePair[] getFilePairsIn(File dir) throws IOException, Exception {
        ArrayList<FilePair> filePairs = new ArrayList<>();
        
        if (!dir.isDirectory())
            throw new IOException();
        
        for (File subDir : dir.listFiles()) {
            if (subDir.isDirectory())
                filePairs.add(new FilePair(subDir.listFiles()));
        }

        FilePair[] pairs = new FilePair[filePairs.size()];
        return filePairs.toArray(pairs);
    }

    public void writeEvalToFile(Evaluation eval, Timer t, String classifierName) throws IOException, Exception {
        File resultsFile = new File(this.resultsDir + classifierName);
        FileWriter writer = new FileWriter(resultsFile);
        writer.write("====  TIME  ====\n" + Double.toString(t.getTime() / Math.pow(10, 9)) + " seconds\n\n");
        writer.write("====  SUMMARY  ====\n" + eval.toSummaryString());
        writer.close();
    }

    public Instances getInstancesFrom(File file) throws Exception {
        DataSource source = new DataSource(new FileInputStream(file));
        return source.getDataSet();
    }

    public static void main(String[] args) {
        //from MLfinalProject dir, run with
        //'java -Dfile.encoding=UTF-8 @/tmp/cp_vngd4z7nsodb9j8f4wl3xwie.argfile ml.App ./arffs/ ./results/'
        if (args.length != 2) {
            args = new String[] {"./arffs/", "./results"};
            System.out.println("WARNING: Defaulting to " + args[0] + " and " + args[1] + " for CLI args.");
        }
        App app = new App(args[0], args[1]);
        app.run();
    }
}