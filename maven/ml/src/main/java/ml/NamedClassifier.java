package ml;

import weka.classifiers.Classifier;

public class NamedClassifier {
    
    private final String name;
    private final Classifier classifier;

    public NamedClassifier(Classifier classifier, String name) {
        this.classifier = classifier;
        this.name = name;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public String toString() {
        return name;
    }
}