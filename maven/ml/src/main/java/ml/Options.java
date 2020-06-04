package ml;

import java.util.Arrays;

public class Options {
    
    private String[] options;

    public Options(String[] options) {
        this.options = Arrays.copyOf(options, options.length);
    }

    public Options (Options copy) {
        this.options = Arrays.copyOf(copy.options, copy.options.length);
    }

    public String[] getOptions() {
        return Arrays.copyOf(options, options.length);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        
        for (String option : this.options) {
            sb.append(option);
            sb.append(",");
        }

        sb.deleteCharAt(sb.length() - 1);

        return sb.toString();
    }
}