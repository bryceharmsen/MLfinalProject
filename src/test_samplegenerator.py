import unittest
from operator import attrgetter
import copy
import math
from samplegenerator import SampleGenerator, ProtoDict, SamplesDict
import util

NUM_SAMPLES = 1000
MAX_NOISE = 0.20
PROTOTYPES = {
    'A': [0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1],
    'B': [1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0],
    'C': [0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1],
    'D': [1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0],
    'E': [1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1],
    'F': [1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0],
    'G': [0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1],
    'H': [1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1],
    'I': [1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1],
    'J': [1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,1,1,0,0,0]
}

EXAMPLE_TRAINING_DATA_SHAPE = {
    'A': [
        [0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1],
        [0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1],
        [0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1]
    ],
    'B': [
        [1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0],
        [1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,0,1,1,1,1,0,0,0],
        [1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,0,0,0,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,0]
        ],
    'C': [
        [0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1],
        [0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,1,1],
        [0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1]
    ],
    'D': [
        [1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0],
        [1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0],
        [1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0]
    ],
    'E': [
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1],
        [1,1,0,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1]
    ],
    'F': [
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0]
    ],
    'G': [
        [0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1],
        [0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1],
        [0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1]
    ],
    'H': [
        [1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1]
    ],
    'I': [
        [1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1]
    ],
    'J': [
        [1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,1,1,0,0,0],
        [1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,1,1,0,0,0],
        [1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,1,1,0,0,0]
    ]
}

class SampleGeneratorTests(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.prototypes = PROTOTYPES
        self.dg = SampleGenerator(copy.deepcopy(PROTOTYPES), NUM_SAMPLES, MAX_NOISE)

    def testConstructor(self):
        self.assertEqual(NUM_SAMPLES, self.dg.numSamples)
        self.assertEqual(MAX_NOISE, self.dg.maxNoise)
        for target in self.prototypes.keys():
            self.assertListEqual(self.dg.prototypes[target], self.prototypes[target])
    
    def testSizeOfNoisySamples(self):
        for target, bits in self.prototypes.items():
            samples = self.dg.getNoisySamples(bits, self.dg.numSamples, self.dg.maxNoise)
            self.assertEqual(len(samples), self.dg.numSamples)

    def testShapeOfNoisySamples(self):
        for key, protoVal in self.prototypes.items():
            samples = self.dg.getNoisySamples(protoVal, self.dg.numSamples, self.dg.maxNoise)
            self.assertIsInstance(samples, list)
            for sample in samples:
                self.assertIsInstance(sample, list)
    
    def testContentsOfNoisySamples(self):
        for target, bits in self.prototypes.items():
            protoBitSum = sum(bits)
            samples = self.dg.getNoisySamples(bits, self.dg.numSamples, self.dg.maxNoise)
            for sample in samples:
                noisyBitSum = 0
                for bit in sample:
                    noisyBitSum += bit
                    self.assertTrue(bit is 0 or bit is 1)
                variance = math.ceil(len(sample) * self.dg.maxNoise)
                self.assertGreaterEqual(noisyBitSum, protoBitSum - variance)
                self.assertLessEqual(noisyBitSum, protoBitSum + variance)

    def testSizeOfGetSamples(self):
        trainingSamples, testingSamples = self.dg.getSamples(self.prototypes, self.dg.numSamples, self.dg.maxNoise)
        numTrainingSamples = int(self.dg.TRAINING_PERCENTAGE * self.dg.numSamples)
        for samples in trainingSamples.values():
            self.assertEqual(len(samples), numTrainingSamples)
        for samples in testingSamples.values():
            self.assertEqual(len(samples), self.dg.numSamples - numTrainingSamples)

    def testShapeOfGetSamples(self):
        trainingAndTestingSetTuple = self.dg.getSamples(self.prototypes, self.dg.numSamples, self.dg.maxNoise)
        for tSet in trainingAndTestingSetTuple:
            self.assertIsInstance(tSet, dict)
            for char, samples in tSet.items():
                self.assertIsInstance(samples, list)
                for sample in samples:
                    self.assertIsInstance(sample, list)
    
    def testContentsOfGetSamples(self):
        trainingAndTestingSetTuple = self.dg.getSamples(self.prototypes, self.dg.numSamples, self.dg.maxNoise)
        for tSet in trainingAndTestingSetTuple:
            for samples in tSet.values():
                for sample in samples:
                    for bit in sample:
                        self.assertTrue(bit is 0 or bit is 1)


if __name__ == "__main__":
    unittest.main()