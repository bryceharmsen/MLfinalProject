package ml;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;

public class ClassifierRunner {
    
    private final AbstractClassifier classifier;
    private final InstancesPair instancesPair;
    private final Options[] optionsList;
    private Options bestOptions;
    private boolean hasBeenTuned;
    private final String classifierName;

    public ClassifierRunner(AbstractClassifier classifier, InstancesPair instancesPair, Options[] optionsList, String classifierName) {
        this.classifier = classifier;
        this.instancesPair = instancesPair;
        this.optionsList = optionsList;
        this.bestOptions = optionsList[0];
        this.hasBeenTuned = false;
        this.classifierName = classifierName;
    }

    public ClassifierRunner(ClassifierRunner copy) {
        this.classifier = copy.classifier;
        this.instancesPair = copy.instancesPair;
        this.optionsList = copy.optionsList;
        this.bestOptions = copy.bestOptions;
        this.hasBeenTuned = copy.hasBeenTuned;
        this.classifierName = copy.classifierName;
    }

    public Evaluation evaluate(Options options) throws Exception {
        return this.evaluate(this.classifier, this.instancesPair, options);
    }

    private Evaluation evaluate(AbstractClassifier classifier, InstancesPair instancesPair, Options options) throws Exception {
        classifier = this.buildClassifier(classifier, instancesPair, options);

        Evaluation eval = new Evaluation(instancesPair.getTrainingInstances());
        eval.evaluateModel(classifier, instancesPair.getTestingInstances());

        return eval;
    }

    public AbstractClassifier buildClassifier(Options options) throws Exception {
        return this.buildClassifier(this.classifier, this.instancesPair, options);
    }

    private AbstractClassifier buildClassifier(AbstractClassifier classifier, InstancesPair instancesPair, Options options) throws Exception {
        classifier.setOptions(options.getOptions());
        classifier.buildClassifier(instancesPair.getTrainingInstances());
        return classifier;
    }

    public Evaluation getBestEvaluation() throws Exception {
        return this.getBestEvaluation(this.classifier, this.instancesPair, this.optionsList);
    }

    private Evaluation getBestEvaluation(AbstractClassifier classifier, InstancesPair instancesPair, Options[] optionsList) throws Exception {
        this.bestOptions = this.tune(classifier, instancesPair, optionsList);
        return this.evaluate(classifier, instancesPair, this.bestOptions);
    }

    public Options tune() throws Exception {
        return this.tune(this.classifier, this.instancesPair, this.optionsList);
    }

    private Options tune(AbstractClassifier classifier, InstancesPair instancesPair, Options[] optionsList) throws Exception {
        double mostPctCorrect = 0.0;

        for (Options options : optionsList) {
            Evaluation eval = this.evaluate(classifier, instancesPair, options);
            double pctCorrect = eval.pctCorrect();
            if (pctCorrect > mostPctCorrect) {
                mostPctCorrect = pctCorrect;
                this.bestOptions = options;
            }
        }
        
        this.hasBeenTuned = true;

        return this.bestOptions;
    }

    public Options getBestOptions() throws Exception {
        if (!this.hasBeenTuned)
            throw new Exception("Cannot get best options - classifier has not been tuned");
        return this.bestOptions;
    }

    public String toString() {
        return this.classifierName;
    }
}