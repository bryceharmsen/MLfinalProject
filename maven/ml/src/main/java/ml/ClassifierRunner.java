package ml;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class ClassifierRunner {
    
    private final Classifier classifier;
    private final InstancesPair instancesPair;
    private final Options[] optionsList;
    private final String classifierName;

    public ClassifierRunner(Classifier classifier, InstancesPair instancesPair, Options[] optionsList, String classifierName) {
        this.classifier = classifier;
        this.instancesPair = instancesPair;
        this.optionsList = optionsList;
        this.classifierName = classifierName;
    }

    public ClassifierRunner(ClassifierRunner copy) {
        this.classifier = copy.classifier;
        this.instancesPair = copy.instancesPair;
        this.optionsList = copy.optionsList;
        this.classifierName = copy.classifierName;
    }

    public static Options tune(Classifier classifier, InstancesPair instancesPair, Options[] optionsList) throws Exception {
        Options bestOptions = null;
        double mostPctCorrect = 0.0;

        for (Options options : optionsList) {
            Evaluation eval = ClassifierRunner.evaluate(classifier, instancesPair, options);
            double pctCorrect = eval.pctCorrect();
            if (pctCorrect > mostPctCorrect) {
                mostPctCorrect = pctCorrect;
                bestOptions = options;
            }
        }
        
        return bestOptions;
    }

    public Options tune() throws Exception {
        return ClassifierRunner.tune(this.classifier, this.instancesPair, this.optionsList);
    }

    public static Evaluation evaluate(Classifier classifier, InstancesPair instancesPair, Options options) throws Exception {
        classifier = ClassifierRunner.buildClassifier(classifier, instancesPair, options);

        Evaluation eval = new Evaluation(instancesPair.getTrainingInstances());
        eval.evaluateModel(classifier, instancesPair.getTestingInstances());

        return eval;
    }

    public Evaluation evaluate(Options options) throws Exception {
        return ClassifierRunner.evaluate(this.classifier, this.instancesPair, options);
    }

    public static Classifier buildClassifier(Classifier classifier, InstancesPair instancesPair, Options options) {
        classifier.setOptions(options.toString());
        classifier.buildClassifier(instancesPair.getTrainingInstances());
        return classifier;
    }

    public Classifier buildClassifier(Options options) {
        return ClassifierRunner.buildClassifier(this.classifier, this.instancesPair, options);
    }

    public static Evaluation getBestEvaluation(Classifier classifier, InstancesPair instancesPair, Options[] optionsList) throws Exception {
        Options bestOptions = ClassifierRunner.tune(classifier, instancesPair, optionsList);
        return ClassifierRunner.evaluate(classifier, instancesPair, bestOptions);
    }

    public Evaluation getBestEvaluation() throws Exception {
        return ClassifierRunner.getBestEvaluation(this.classifier, this.instancesPair, this.optionsList);
    }

    public String toString() {
        return this.classifierName;
    }
}