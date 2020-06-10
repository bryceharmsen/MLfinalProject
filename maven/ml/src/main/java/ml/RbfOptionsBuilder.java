package ml;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import weka.classifiers.functions.RBFNetwork;

public class RbfOptionsBuilder extends OptionsBuilder {
    
    public RbfOptionsBuilder(int numGeneratedOptions) throws Exception {
        super(new RBFNetwork(), numGeneratedOptions);
    }

    public RbfOptionsBuilder(RbfOptionsBuilder copy) {
        super(copy);
    }

    protected ArrayList<Options> generateOptions(int numGeneratedOptions) throws Exception {
        ArrayList<Options> generatedOptions = new ArrayList<>();
        RBFNetwork rbf = (RBFNetwork) this.classifier;
        rbf.setNumClusters(10);
        String[] defaultOptions = this.classifier.getOptions();

        //Iterator<Integer> numClusters = (new Random()).ints(50, 501).iterator();
        Iterator<Double> ridge = (new Random()).doubles(0.1, 0.4).iterator();
        Iterator<Double> minStdDev = (new Random()).doubles(0.01, 10).iterator();
        
        while(numGeneratedOptions-- > 0) {
            rbf.setOptions(defaultOptions);
            rbf.setMinStdDev(minStdDev.next());
            rbf.setRidge(ridge.next());
            //options[1] = "10";//Integer.toString(numClusters.next());
            //options[5] = Double.toString(ridge.next());
            //options[9] = Double.toString(minStdDev.next());

            generatedOptions.add(new Options(rbf.getOptions()));
        }

        return generatedOptions;
    }

}