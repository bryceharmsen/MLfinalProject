package ml;

import java.io.File;
import java.io.FileWriter;
import java.io.FileInputStream;
import java.io.InputStream;
import java.lang.StackWalker.Option;
import java.io.IOException;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.functions.LibSVM;

public class App {

    private String arffDir;
    private String resultsDir;

    public App(String arffDir, String resultsDir) {
        this.arffDir = arffDir;
        this.resultsDir = resultsDir;
    }


    public void run() throws IOException, Exception {
        FilePair[] filePairs = this.getFilePairsIn(new File(this.arffDir));
        NamedClassifier[] classifiers = {
            new NamedClassifier(new MultilayerPerceptron(), "MLP"),
            new NamedClassifier(new SVM(), "SVM"),
            new NamedClassifier(new RBF(), "RBF"),
            new NamedClassifier(new RandomForest(), "RandomForest")
        };
        for (FilePair pair : filePairs) {
            InstancesPair instancesPair = new InstancesPair(this.getInstanceFrom(pair.getTrainingFile()),
                                                 this.getInstanceFrom(pair.getTestingFile()));
            for (NamedClassifier classifier : classifiers) {
                //TODO finish OptionsBuilder
                OptionBuilder builder = new OptionBuilder();
                ClassifierRunner runner = new ClassifierRunner(classifier.getClassifier(), instancesPair, builder.getOptions(), classifier.toString());
                Timer t = new Timer();
                t.start();
                Evaluation eval = runner.getBestEvaluation();
                t.end();
                //report results to file
                this.writeEvalToFile(eval, classifier.toString());
            }
        }
    }

    public void runs() throws IOException, Exception {
        //get names of arff files
        //get sets of train and test data from arff files
        FilePair[] filePairs = this.getFilePairsIn(new File(this.arffDir));
        //build array of mlp, svm, rbf, randomForest classifiers
        ClassifierRunner[] classifiers = {};
        //for each set of train and test data
        for (FilePair pair : filePairs) {
            Instances trainData = this.getInstanceFrom(pair.getTrainingFile());
            Instances testData = this.getInstanceFrom(pair.getTestingFile());
            //for each classifier
            for (Classifier classifier : classifiers) {
                //tune classifier
                String[] bestOptions = this.tuneClassifier(classifier, this.getOptionsSets(classifier), trainData, testData);
                Evaluation eval = this.getEvaluation(classifier, bestOptions, trainData, testData);
                //save results from classifier in file
                this.writeEvalToFile(eval, this.getAssociatedFileName(classifier));
            }
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

    public void writeEvalToFile(Evaluation eval, String classifierName) throws IOException {
        File resultsFile = new File(this.resultsDir + classifierName);
        FileWriter writer = new FileWriter(resultsFile);
        writer.write(eval.toSummaryString());
        writer.close();
    }

    public Instances getInstanceFrom(File file) throws Exception {
        DataSource source = new DataSource(new FileInputStream(file));
        return source.getDataSet();
    }

    public double executeTrial(Classifier classifier, String[] options, String resultsFilePath, String trainingFilePath, String testingFilePath) throws Exception, IOException {
        System.out.println("Results will go to " + resultsFilePath + ".txt");
        File resultsFile = new File(resultsFilePath + ".txt");
        FileWriter writer = new FileWriter(resultsFile);

        DataSource trainSource = new DataSource(trainingFilePath),
                testSource = new DataSource(testingFilePath);
        Instances trainData = trainSource.getDataSet(),
                testData = testSource.getDataSet();
        writer.write(
            trainData.numInstances() + " training instances loaded.\n" +
            testData.numInstances() + " testing instances loaded.\n"
        );
        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(testData.numAttributes() - 1);

        classifier.setOptions(options);
        classifier.buildClassifier(trainData);
        writer.write(classifier.toString());

        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);
        writer.write(eval.toSummaryString("Results\n", false));
        writer.close();

        return eval.pctCorrect();
    }
    public static void main(String[] args) throws Exception {
        //from MLfinalProject dir, run with 'java -Dfile.encoding=UTF-8 @/tmp/cp_vngd4z7nsodb9j8f4wl3xwie.argfile ml.App ./arffs/ ./results/'
        MyMLP mlp = new MyMLP(args[0], args[1]);
        mlp.run();
    }
}