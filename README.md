# MLfinalProject
Individual Final Project for Machine Learning 457 at CWU.

## Java

To run the Maven Java project, open the maven directory in a Java-compliant IDE and run through the IDE. The author ran this code using VS Code and the Maven for Java extension. If run from the command line, the required CLI args are
```<arff directory path> <results directory path> <log path and filename to create> <number of tuning trials>```
These paths can be relative from the user's current directory. The number of tuning trials defines the number of iterations to perform random search for hyperparameter optimization for each configuration of each ML algorithm.

## Python

To run the Python sample generator, change directory to the python/src/ directory and run:  
```python3 samplegeneratory.py ../data/prototypes.yaml ../config```