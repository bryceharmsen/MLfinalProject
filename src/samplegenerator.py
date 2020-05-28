#for characters A through J
    #create one 7x7 prototype
    #binary matrix
#represent matrices as vectors

#should the prototypes go in a file that
#can be read?

#for each prototype
    #generate n_samples - 1 noisy variations
    #invert max_noise percent of bits
#include 10 prototypes and data set
#of 10*(n_samples) vectors
#randomly divide data set into
#60% training
#40% testing

#train/test on with classifiers
#MLP
#RBF
#SVM
#Random Forest

#compare results by classification accuracy

#for 3 different values of
#n_samples and max_noise
# repeat previous experiment and
# observe behavior of classifiers

#decide which classifier performs best
#on average for the given problem
#consider both accuracy and training time
#(PUT A TIMER ON TRIALS)

#I need:
#[ ] 1. Timer around trials
#[x] 2. 3 different n_sample and max_noise combos
#[x] 3. 10 Character prototypes
#[x] 4. noise generator that uses n_samples and max_noise
#[x] 5. random data set divider function (shuffle and split by percentage)
#[ ] 6. Library implementations of MLP, RBF, SVM, Random Forest
#[ ] 7. data shaper for results, way to easily track accuracy
#[ ] 8. hyperparameter tuner (grid search?) for each ML algorithm
import sys
import random
import math
import copy
import shutil
import util
from typing import List, Dict, Tuple 

STARTER_ARFF_PATH = '../arffs/starter.arff'

ProtoDict = Dict[str, List[int]]
SamplesDict = Dict[str, List[List[int]]]

class SampleGenerator:

    def __init__(self, prototypes: ProtoDict, numSamples: int, maxNoise: int, **kwargs):
        self.prototypes = prototypes
        self.numSamples = numSamples
        self.maxNoise = maxNoise
        self.TRAINING_PERCENTAGE = 0.6
    
    def getSamples(self, prototypes: ProtoDict, numSamples: int, maxNoise: int) -> Tuple[SamplesDict, SamplesDict]:
        trainingSamples = copy.deepcopy(prototypes)
        testingSamples = copy.deepcopy(prototypes)
        for char, bits in prototypes.items():
            noisySamples = self.getNoisySamples(bits,numSamples, maxNoise)
            random.shuffle(noisySamples)
            splitIdx = math.ceil(numSamples * self.TRAINING_PERCENTAGE)
            trainingSamples[char] = noisySamples[:splitIdx]
            testingSamples[char] = noisySamples[splitIdx:]
        return (trainingSamples, testingSamples)

    def getNoisySamples(self, proto: List[int], numSamples: int, maxNoise: int) -> List[List[int]]:
        samples = [[bit for bit in proto]]
        for _ in range(numSamples - 1):
            #copy original proto
            noisySample = [bit for bit in proto]
            for _ in range(int(len(proto) * maxNoise)):
                #if coin flip chooses to flip bit
                if random.choice([0,1]):
                    #flip random bit
                    idx = int(random.uniform(0, len(proto)))
                    noisySample[idx] = abs(noisySample[idx] - 1)
            samples.append(noisySample)
        return samples

    def getArffs(self, saveDir='.', filePrefix=''):
        trainingSamples, testingSamples = self.getSamples(self.prototypes, self.numSamples, self.maxNoise)
        trainFileName = f'{filePrefix}train'
        testFileName = f'{filePrefix}test'
        self.generateArff(trainingSamples, saveDir, trainFileName)
        self.generateArff(testingSamples, saveDir, testFileName)
        print(f'{trainFileName} and {testFileName} have been saved at {saveDir}')


    def generateArff(self, samplesDict: SamplesDict, saveDir='.', fileName='data'):
        saveDir = saveDir[:-1] if saveDir.endswith('/') else saveDir
        savePath = f'{saveDir}/{fileName}.arff'
        shutil.copy2(STARTER_ARFF_PATH, savePath)
        with open(savePath, 'a') as arff:
            for char, samples in samplesDict.items():
                for sample in samples:
                    bitStr = ''
                    for bit in sample:
                        bitStr += f'{bit},'
                    arff.write(f'{bitStr}{char}\n')
    

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print(f'CLI Use:\n\tpython3 {sys.argv[0]} <PATH_TO_PROTOTYPES_FILE.yaml> <PATH_TO_CONFIG_DIR>')
        sys.exit(1)
    prototypes = util.parseYaml(sys.argv[1])
    paramsByFile = util.parseYamlsFrom(sys.argv[2])
    sampleSetsByParams = []
    for params in paramsByFile:
        print(params)
        dg = SampleGenerator(prototypes, **params)
        dg.getArffs('../arffs', params['arffName'])
        samples = dg.getSamples(dg.prototypes, dg.numSamples, dg.maxNoise)
        sampleSetsByParams.append(samples)