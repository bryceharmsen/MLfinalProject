package ml;

import weka.core.Instances;

public class InstancesPair {
    
    private final Instances training;
    private final Instances testing;

    public InstancesPair(Instances training, Instances testing) {
        this.training = training;
        this.testing = testing;

        training.setClassIndex(training.numAttributes() - 1);
        testing.setClassIndex(testing.numAttributes() - 1);
    }

    public InstancesPair(InstancesPair copy) {
        this.training = copy.training;
        this.testing = copy.testing;
    }

    public Instances getTrainingInstances() {
        return this.training;
    }

    public Instances getTestingInstances() {
        return this.testing;
    }
}