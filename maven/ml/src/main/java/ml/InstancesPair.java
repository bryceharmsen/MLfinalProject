package ml;

import weka.core.Instances;

public class InstancesPair {
    
    private final String dirName;
    private final Instances training;
    private final Instances testing;

    public InstancesPair(Instances training, Instances testing, String dirName) {
        this.dirName = dirName;
        this.training = training;
        this.testing = testing;

        training.setClassIndex(training.numAttributes() - 1);
        testing.setClassIndex(testing.numAttributes() - 1);
    }

    public InstancesPair(InstancesPair copy) {
        this.dirName = copy.dirName;
        this.training = copy.training;
        this.testing = copy.testing;
    }

    public String getDirName() {
        return this.dirName;
    }

    public Instances getTrainingInstances() {
        return this.training;
    }

    public Instances getTestingInstances() {
        return this.testing;
    }
}