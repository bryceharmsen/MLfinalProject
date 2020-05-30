package ml;

public class OptionBuilder {
    
    private Options defaultOptions;

    public OptionBuilder(String[] defaultOptions) {
        this.defaultOptions = new Options(defaultOptions);
    }

    public Options getOptions() {
        return this.defaultOptions;
    }
}