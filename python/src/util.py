import os
import yaml

def parseYaml(fileName: str) -> dict:
    with open(fileName) as file:
        return yaml.full_load(file)

def getFilesFrom(dirPath: str, containing=''):
    chosenFileNames = []
    for (path, dirNames, fileNames) in os.walk(dirPath):
        for fileName in fileNames:
            if fileName.find(containing) != -1:
                chosenFileNames.append(fileName)
        break
    return chosenFileNames

def parseYamlsFrom(dirPath: str, containing=''):
    dirPath = dirPath[:-1] if dirPath.endswith('/') else dirPath
    fileNames = getFilesFrom(dirPath, containing='yaml')
    parsedDicts = []
    for fileName in fileNames:
        parsedDicts.append(parseYaml(f'{dirPath}/{fileName}'))
    return parsedDicts