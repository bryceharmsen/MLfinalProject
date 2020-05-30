package ml;

import weka.classifiers.AbstractClassifier;

public class NamedClassifier {
    
    private final String name;
    private final AbstractClassifier classifier;

    public NamedClassifier(AbstractClassifier classifier, String name) {
        this.classifier = classifier;
        this.name = name;
    }

    public AbstractClassifier getClassifier() {
        return this.classifier;
    }

    public String getName() {
        return this.name;
    }
}