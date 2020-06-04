package ml;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import weka.classifiers.trees.RandomForest;

public class RandForestOptionsBuilder extends OptionsBuilder {
    
    public RandForestOptionsBuilder(int numGeneratedOptions) {
        super(new RandomForest(), numGeneratedOptions);
    }

    public RandForestOptionsBuilder(RandForestOptionsBuilder copy) {
        super(copy);
    }

    protected ArrayList<Options> generateOptions(int numGeneratedOptions) {
        ArrayList<Options> generatedOptions = new ArrayList<>();
        String[] options = this.classifier.getOptions();

        Iterator<Integer> numTrees = (new Random()).ints(10, 300).iterator();
        Iterator<Integer> numFeatures = (new Random()).ints(1, 10).iterator();
        
        while(numGeneratedOptions-- > 0) {
            options[1] = Integer.toString(numTrees.next());
            options[3] = Integer.toString(numFeatures.next());

            generatedOptions.add(new Options(options));
        }

        return generatedOptions;
    }
}