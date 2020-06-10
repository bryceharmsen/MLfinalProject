package ml;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import weka.classifiers.trees.RandomForest;

public class RandForestOptionsBuilder extends OptionsBuilder {
    
    public RandForestOptionsBuilder(int numGeneratedOptions) throws Exception {
        super(new RandomForest(), numGeneratedOptions);
    }

    public RandForestOptionsBuilder(RandForestOptionsBuilder copy) throws Exception {
        super(copy);
    }

    protected ArrayList<Options> generateOptions(int numGeneratedOptions) throws Exception {
        ArrayList<Options> generatedOptions = new ArrayList<>();
        RandomForest rf = (RandomForest) this.classifier;
        String[] defaultOptions = this.classifier.getOptions();

        Iterator<Integer> numFeatures = (new Random()).ints(1, 10).iterator();
        Iterator<Integer> bagSizePercent = (new Random()).ints(5, 100).iterator();
        
        while(numGeneratedOptions-- > 0) {
            rf.setOptions(defaultOptions);
            rf.setNumFeatures(numFeatures.next());
            rf.setBagSizePercent(bagSizePercent.next());
            //options[3] = Integer.toString(numTrees.next());
            //options[3] = Integer.toString(numFeatures.next());

            generatedOptions.add(new Options(rf.getOptions()));
        }

        return generatedOptions;
    }
}