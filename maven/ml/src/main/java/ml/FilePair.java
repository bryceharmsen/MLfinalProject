package ml;

import java.io.File;

public class FilePair {
    
    private final String dirName;
    private final File training;
    private final File testing;

    public FilePair(File[] filePair, String dirName) throws Exception {
        this.dirName = dirName;
        if (filePair[0].getName().contains("train")) {
            this.training = filePair[0];
            this.testing = filePair[1];
        } else if (filePair[1].getName().contains("train")) {
            this.training = filePair[1];
            this.testing = filePair[0];
        } else {
            throw new Exception("Could not identify training set.");
        }
    }

    public FilePair(FilePair copy) {
        this.dirName = copy.dirName;
        this.training = copy.training;
        this.testing = copy.testing;
    }

    public String getDirName() {
        return this.dirName;
    }

    public File getTrainingFile() {
        return this.training;
    }

    public File getTestingFile() {
        return this.testing;
    }
}