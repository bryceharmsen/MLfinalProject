package ml;

import java.util.Iterator;
import java.util.Random;
import java.util.ArrayList;

import weka.classifiers.functions.MultilayerPerceptron;

public class MlpOptionsBuilder extends OptionsBuilder {

    public MlpOptionsBuilder(int numGeneratedOptions) {
        super(new MultilayerPerceptron(), numGeneratedOptions);
    }

    public MlpOptionsBuilder(MlpOptionsBuilder copy) {
        super(copy);
    }

    protected ArrayList<Options> generateOptions(int numGeneratedOptions) {
        ArrayList<Options> generatedOptions = new ArrayList<>();
        String[] options = this.classifier.getOptions();

        Iterator<Double> learningRate = (new Random()).doubles(0.1, 0.4).iterator();
        Iterator<Integer> trainingTime = (new Random()).ints(50, 501).iterator();
        Iterator<Integer> neurons = (new Random()).ints(1, 20).iterator();
        Iterator<Integer> layers = (new Random()).ints(1, 3).iterator();
        
        while(numGeneratedOptions-- > 0) {
            options[1] = Double.toString(learningRate.next());
            options[5] = Integer.toString(trainingTime.next());
            options[13] = this.getNeuronConfiguration(neurons, layers);

            generatedOptions.add(new Options(options));
        }

        return generatedOptions;
    }

    private String getNeuronConfiguration(Iterator<Integer> neurons, Iterator<Integer> layers) {
        StringBuilder config = new StringBuilder();
        int numLayers = layers.next();

        for (int i = 0; i < numLayers; i++) {
            config.append(neurons.next());
            config.append(i < numLayers - 1 ? "," : "");
        }

        return config.toString();
    }

}