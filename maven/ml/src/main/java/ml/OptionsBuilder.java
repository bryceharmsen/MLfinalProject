package ml;

import java.util.ArrayList;

import weka.classifiers.AbstractClassifier;

public abstract class OptionsBuilder {
    
    protected AbstractClassifier classifier;
    protected Options defaultOptions;
    protected ArrayList<Options> generatedOptions;
    protected int numGeneratedOptions;

    public OptionsBuilder(AbstractClassifier classifier, int numGeneratedOptions) throws Exception {
        this.classifier = classifier;
        this.defaultOptions = new Options(classifier.getOptions());
        this.numGeneratedOptions = numGeneratedOptions;
        this.generatedOptions = this.generateOptions(numGeneratedOptions);
    }

    public OptionsBuilder(OptionsBuilder copy) {
        this.classifier = copy.classifier;
        this.defaultOptions = copy.defaultOptions;
        this.numGeneratedOptions = copy.numGeneratedOptions;
        this.generatedOptions = copy.generatedOptions;
    }

    public Options getDefaultOptions() {
        return this.defaultOptions;
    }

    protected abstract ArrayList<Options> generateOptions(int numGeneratedOptions) throws Exception;

    public Options[] getGeneratedOptions() {
        return this.generatedOptions.toArray(new Options[0]);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Options options : this.generatedOptions) {
            sb.append("[");
            sb.append(options.toString());
            sb.append("]\n");
        }

        return sb.toString();
    }
}