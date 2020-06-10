package ml;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import weka.classifiers.functions.SMO;

public class SvmOptionsBuilder extends OptionsBuilder {
    
    public SvmOptionsBuilder(int numGeneratedOptions) throws Exception {
        super(new SMO(), numGeneratedOptions);
    }

    public SvmOptionsBuilder(SvmOptionsBuilder copy) {
        super(copy);
    }

    protected ArrayList<Options> generateOptions(int numGeneratedOptions) throws Exception {
        ArrayList<Options> generatedOptions = new ArrayList<>();
        SMO smo = (SMO) this.classifier;
        String[] defaultOptions = this.classifier.getOptions();

        Iterator<Double> c = (new Random()).doubles(0.1, 10).iterator();
        Iterator<Double> tolerance = (new Random()).doubles(0.1, 0.4).iterator();
        Iterator<Double> epsilon = (new Random()).doubles(0.01, 10).iterator();
        
        while(numGeneratedOptions-- > 0) {
            smo.setOptions(defaultOptions);
            smo.setC(c.next());
            smo.setToleranceParameter(tolerance.next());
            smo.setEpsilon(epsilon.next());
            //options[1] = Double.toString(c.next());
            //options[3] = Double.toString(tolerance.next());
            //options[5] = Double.toString(epsilon.next());

            generatedOptions.add(new Options(smo.getOptions()));
        }

        return generatedOptions;
    }

}