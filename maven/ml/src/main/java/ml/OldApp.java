package ml;

import java.io.File;
import java.io.FileWriter;
import java.io.FileInputStream;
import java.io.InputStream;
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

public class OldApp {

    private String arffPath;

    public OldApp(String arffPath) {
        this.arffPath = arffPath;
    }

    public void run() throws IOException, Exception {
        //get names of arff files
        //get sets of train and test data from arff files
        File[][] arffFilePairs = this.getFilePairsIn(this.arffPath);
        //build array of mlp, svm, rbf, randomForest classifiers
        Classifier[] classifiers = {};
        //for each set of train and test data
        for (File[] pair : arffFilePairs) {
            Instances trainData = this.getInstanceFrom(pair[0]);
            Instances testData = this.getInstanceFrom(pair[1]);
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

    public String[][] getOptionsSets(Classifier classifier) {
        String[][] optionsSets = {};
        return optionsSets;
    }

    public String getAssociatedFileName(Classifier classifier) {
        String fileName = "";
        return fileName;
    }

    public File[][] getFilePairsIn(String dirPath) throws IOException {
        ArrayList<File[]> filePairs = new ArrayList<>();
        File dir = new File(dirPath);
        
        if (!dir.isDirectory())
            throw new IOException();
        
        for (File subDir : dir.listFiles()) {
            if (subDir.isDirectory())
                filePairs.add(subDir.listFiles());
        }

        File[][] pairs = new File[filePairs.size()][filePairs.get(0).length];
        for (int i = 0; i < pairs.length; i++) {
            pairs[i] = filePairs.get(i);
        }
        return pairs;
    }

    public void writeEvalToFile(Evaluation eval, String filePath) throws IOException {
        File resultsFile = new File(filePath);
        FileWriter writer = new FileWriter(resultsFile);
        writer.write(eval.toSummaryString());
        writer.close();
    }

    public Instances getInstanceFrom(File file) throws Exception {
        DataSource source = new DataSource(new FileInputStream(file));
        return source.getDataSet();
    }

    public String[] tuneClassifier(Classifier classifier, String[][] optionsSets, Instances trainData, Instances testData) throws Exception {
        double mostPctCorrect = 0;
        String[] bestOptions = optionsSets[0];
        
        for (String[] options : optionsSets) {
            Evaluation eval = this.getEvaluation(classifier, options.clone(), trainData, testData);
            double currPctCorrect = eval.pctCorrect();
            if (currPctCorrect > mostPctCorrect) {
                mostPctCorrect = currPctCorrect;
                bestOptions = options;
            }
        }
        
        return bestOptions;
    }

    public void buildClassifier(Classifier classifier, String[] options, Instances trainData, Instances testData) throws Exception {
        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(testData.numAttributes() - 1);
        
        classifier.setOptions(options);
        classifier.buildClassifier(trainData);
    }

    public Evaluation getEvaluation(Classifier classifier, String[] options, Instances trainData, Instances testData) throws Exception {
        this.buildClassifier(classifier, options, trainData, testData);

        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);

        return eval;
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
        //from MLfinalProject dir, run with 'java -Dfile.encoding=UTF-8 @/tmp/cp_vngd4z7nsodb9j8f4wl3xwie.argfile ml.OldApp ./arffs/ ./results/'
        MyMLP mlp = new MyMLP(args[0], args[1]);
        mlp.run();
    }
}