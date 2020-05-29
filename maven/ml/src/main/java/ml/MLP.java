package ml;

import weka.core.Instances;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;

public class MLP {
    private String resultsPath;
    private String trainingSamplesFile;
    private String testingSamplesFile;

    public MLP(String arffPath, String resultsPath) {
        this.trainingSamplesFile = arffPath + "Atrain.arff";
        this.testingSamplesFile = arffPath + "Atest.arff";
        this.resultsPath = resultsPath;
    }

    public double executeTrial(String[] options, String fileName) throws Exception, IOException {
        System.out.println(fileName);
        File resultsFile = new File(this.resultsPath + fileName + ".txt");
        FileWriter writer = new FileWriter(resultsFile);

        DataSource trainSource = new DataSource(this.trainingSamplesFile),
                testSource = new DataSource(this.testingSamplesFile);
        Instances trainData = trainSource.getDataSet(),
                testData = testSource.getDataSet();
        writer.write(
            trainData.numInstances() + " training instances loaded.\n" +
            testData.numInstances() + " testing instances loaded.\n"
        );
        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(testData.numAttributes() - 1);

        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setOptions(options);
        mlp.buildClassifier(trainData);
        writer.write(mlp.toString());
        mlp.setGUI(true);

        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(mlp, testData);
        writer.write(eval.toSummaryString("Results\n", false));
        writer.close();

        return eval.pctCorrect();
    }

    public String[] getNeuronConfigurations(int loNeuron, int hiNeuron, int loLayers, int hiLayers) {
        ArrayList<String> configs = new ArrayList<>();
        Random rand = new Random();
        Iterator<Integer> ints = rand.ints(loNeuron, hiNeuron + 1).iterator();
        for (int i = loLayers; i <= hiLayers; i++) {
            for (int j = loNeuron; j <= hiNeuron; j++) {
                StringBuilder str = new StringBuilder();
                for (int k = 0; k < i; k++) {
                    str.append(ints.next());
                    if (k < i - 1) str.append(",");
                }
                configs.add(str.toString());
            }
        }
        return configs.toArray(new String[0]);
    }

    public void run() throws Exception {
        try {
            //0.3 learning rate
            //0.2 momentum
            //10 epochs
            String[] options = {
                "-L", "0.3", "-M", "0.2",
                "-N", "5000", "-V", "0", 
                "-S", "0", "-E", "20", 
                "-H", "1"
            };
            int minNeuron = 0, maxNeuron = 19,
                minLayer = 1, maxLayer = 2;
            String[] configs = this.getNeuronConfigurations(minNeuron, maxNeuron, minLayer, maxLayer);
            double mostCorrect = -1;
            String bestFileName = "NO FILE CREATED";
            for (String config : configs) {
                options[options.length - 1] = config;
                String fileName = config.replace(",", "-");
                double correct = 0;
                try {
                    correct = this.executeTrial(Arrays.copyOf(options, options.length), fileName);
                } catch (IOException ioe) {
                    System.out.println("IOException: " + ioe.getMessage() + ". May have already run trial for neuron structure " + config + ", moving on.");
                    continue;
                }
                if (correct > mostCorrect) {
                    mostCorrect = correct;
                    bestFileName = fileName;
                }
            }
            System.out.println("Best results:\n\tMost correct: " + mostCorrect + "\n\tFile: " + bestFileName + ".");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            System.out.println(e.getStackTrace()); 
        }
    }
}