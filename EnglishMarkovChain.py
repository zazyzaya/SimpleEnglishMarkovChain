import random
import re
import os

WORD_COUNT = -1
VOCAB_FILE = 'Ch9.txt'

vocabulary = []
wordToIndex = {}

non_letters = re.compile(r"[^a-zA-Z\s]")

with open(VOCAB_FILE, 'r') as f:
    og = f.read()
    vocabulary = list(set(og.split())) # f: int -> String

with open('out.txt', 'w+') as f:
    f.write(og)

SIZE_OF_VOCAB = len(vocabulary)

for i in range(SIZE_OF_VOCAB):      # g:String -> int
    wordToIndex[vocabulary[i]] = i


class WordNodes:
    """ The WordNodes class is a collection of references to other WordNodes classes
        It is essentially a directed graph where WordNodes are the vectors and the weights
        at relations[i] is the weight of the edge between this wordnode and wordnode i

        name : str
            The string representation of the node

        relations : list
            a list of floats representing the weight of the edge between this node and the node at vocab[i]

        newWords : list
            a list of words detected in the most recent training that were not previously in the vocabulary
            but are related to this node

        old_relations : list
            a list of floats representing the weight of the edge between this node and the node at vocab[i]
            prior to whatever training occurs on this node so they may be updated with new values given different data
    """

    def __init__(self, name, arr=None):
        """ Constructor for the WordNodes class 

            name : str
                The string representation of the node
        """
        self.name = name
        self.relations = [0] * SIZE_OF_VOCAB     # Initializes empty int[850]
        self.newWords = []

        if arr != None:
            self.old_relations = arr
            if len(self.old_relations) < SIZE_OF_VOCAB:
                self.old_relations += [0] * (SIZE_OF_VOCAB - len(self.old_relations))   # Updates length of old list
        else:
            self.old_relations = []

    def normalize(self, size):
        """ This method converts number of references to other vectors to floats between 0 and 1
            building the weight of the edge

            size : int
                The number of other vectors (vocab words)
        """
        for i in range(len(self.relations)):
            self.relations[i] /= size
        
        if self.old_relations != []:
            for i in range(len(self.relations)):
                self.relations[i] = (self.relations[i] + self.old_relations[i])/2

        self.old_relations = self.relations     # In case train is run more than once, this ensures it updates

    def train(self, input):
        """ This method builds the associations between this node and others
            Updates the edges between self and all others

            input : str
                The data this node is being trained on
        """
        global SIZE_OF_VOCAB

        captureNext = False
        numHits = 0

        for testWord in input.split():
            
            if captureNext:
                if testWord not in wordToIndex:
                    self.newWords.append(testWord)

                else: 
                    self.relations[wordToIndex[testWord]] += 1
                    numHits += 1

                captureNext = False

            if testWord == self.name:
                captureNext = True

        if numHits != 0:
            self.normalize(numHits)


def threadBuildWeights(w, trainingData):
    """ An auxillery method to build chain. Originally implimented for multithreading
        (unsuccessfully) kept in because I'm too lazy to merge it back into buildChain()

        w : str
            The string representation of a word to be turned into a node

        trainingData : str  
            The data the word will be trained on

        Returns
        WordNodes
            A node representing the word in the markov chain
    """
    word = WordNodes(w)
    trainingData = re.sub(non_letters, '', trainingData)

    word.train(trainingData)
        #print("Scanning " + self.word.name)

    return word

def updateVocab(newWords, fName='out.txt'):
    """ Updates the markov chain's vocabulary (global variables and file)

        newWords : list
            A list of strings that are not currently in the markov chain's vocabulary

        fName : str
            The name of the file where this will be saved. Default 'out.txt'
    """

    global vocabulary, SIZE_OF_VOCAB

    newVocab = set(newWords)
    if newWords != []:
        with open(fName, 'a') as f:
            for newW in newVocab:
                f.write('\n' + newW)

    vocabulary += newVocab
    SIZE_OF_VOCAB = len(vocabulary)
    for i in range(SIZE_OF_VOCAB - len(newVocab)): # Update vocab dictionary
        wordToIndex[vocabulary[i]] = i

def buildChain(trainingData, updateV=True):
    """ Driver method to build a markov chain from a set of training data

        trainingData : str
            The data the markov chain will learn from 

        updateV : boolean
            If the markov chain should absorb new vocabulary words, this is set to true
            otherwise, it should be set to false. Defaults to true, as usually you want
            new vocabulary at least on the first pass (otherwise stuck in basic 850 english words)

        Returns
        list
            The markov chain that has been loaded into memory
    """
    nodeList = []   # f:Word -> (N X W) where N is the index of the vocab word, and W is the weight of the pointer
    newVocab = []

    for word in vocabulary:
        w = threadBuildWeights(word, trainingData)
        nodeList.append(w.relations)
        
        if w.newWords != []:
            newVocab += w.newWords

    if updateV:
        updateVocab(newVocab)
    
    return nodeList

def updateChain(trainingData, oldNodeList, updateV=True):
    """ Retrains the markov chain by going over the training data again. Useful if new words were found in the last
        pass. Will update vocabulary when words have associations with unknown words unless updateV is false

        trainingData : str
            The input string to train the chain with

        oldNodeList : list
            The markov chain to be updated with this new data

        updateV : boolean
            If during the retraining, the user wants new words added to the vocabulary, this should
            be set to true. Otherwise, it should be set to fault. Default option is true

        Returns
        list
            The updated markov chain
    """

    nodeList = []   # f:Word -> (N X W) where N is the index of the vocab word, and W is the weight of the pointer
    newWords = []

    i = 0
    for w in vocabulary: # O(N^2) :(
        word = WordNodes(w, oldNodeList[i])
        word.train(trainingData)
            
        nodeList.append(word.relations)
        newWords += word.newWords

    if updateV:
        updateVocab(newWords)
    
    return nodeList

def generateSentence(weights, length=50, seed=None):
    """ Really the only point of building a markov chain, this uses it to construct a sentence

        weights : list
            The markov chain being used to generate this sentence
        
        length : int
            The length of the sentence generated. Defaults to 50

        seed : str
            Seeds the markov chain to start on this word (NOTE: undefined behaviour when word not in vocab)

        Returns
        str
            The string generated by the markov chain
    """
    i = 0
    sentence = ''
    
    if not seed:
        lastWord = random.randint(0, len(vocabulary)-1)
    else:
        sentence += seed + ' '
        lastWord = wordToIndex[seed]
        i += 1
    
    while(i < length):
        vector = random.random()
        total = 0
        possibleNext = 0

        while (total + weights[lastWord][possibleNext % (SIZE_OF_VOCAB)] < vector):
            total += weights[lastWord][possibleNext % (SIZE_OF_VOCAB)] + 0.00001   #just in case there are no occurrences of this word
            possibleNext += 1


        sentence += vocabulary[possibleNext % SIZE_OF_VOCAB] + " "
        lastWord = possibleNext % SIZE_OF_VOCAB
        i += 1

    return sentence

def loadChain(fName='BasicEnglishMarkovChain.data'):
    """ Loads a saved markov chain into memory (saves the time of training a new one every time)

        fName : str
            The filename of the markov chain. Defaults to 'BasicEnglishMarkovChain.data'

        Returns
        list
            The markov chain that has been loaded into memory
    """

    nodeList = []
    with open(fName, 'r') as f:
        s = f.read()

    lines = s.split('\n')
    for line in lines:
        innerList = []
        words = line.split(' ')
        for word in words[:-1]:     # Always has a trailing space
            innerList.append(float(word))
            
        nodeList.append(innerList)

    return nodeList

def saveChain(nodeList, fName='BasicEnglishMarkovChain'):
    """ Saves a markov chain and its associated vocabulary so that it can be 
        loaded later using the loadChain() function

        nodeList : list
            The markov chain datastructure

        fName : str
            Optional argument, the file name the chain and its vocabulary will be saved to
    """

    from io import StringIO
    fStr = StringIO()

    print('Saving...')
    for word in nodeList:
        for weight in word:
            fStr.write(str(weight) + ' ')
        fStr.write('\n')

    # Saves chain
    with open(fName + '.data', 'w+') as f:
        f.write(fStr.getvalue())
    
    # Saves vocabulary
    with open(fName+'Vocab.txt', 'w+') as f:
        for word in vocabulary:
            f.write(word + "\n")

def update(nodelist, dataset, updateV=True):
    """ This method is to update an already existing markov chain 
        nodelist : list
             a markov chain generated using makeNew()
        dataset : str
            a string to train the markov chain with
        updateV : boolean
            optional argument to update vocabulary 

        Returns
        list
            The updated markov chain 
    """
    nodeList = loadChain()
    nodeList = updateChain(dataset, nodeList, updateV=updateV)
    return nodeList

def makeNew(dataset):
    """ This method is to build a totally new network just from an input (a book, a website, etc.)

        dataset : str
            a string to train the markov chain with

        Returns
        list
            A new markov chain (datastructure is a 2d list of nodes)
    """
    nodeList = buildChain(dataset)
    return nodeList

def main():
    # Users should set these variables to manipulate the output
    VOCAB_TRAINING = 1      # Only ever needs to be at 1 if vocab == training data
    GRAMMAR_TRAINING = 9
    LEN_OUTPUT = 3000
    TRAINING_FILE = VOCAB_FILE

    dataset = ''
    with open(TRAINING_FILE, 'r') as f: # Input file it will learn from
        dataset = f.read()

    nodelist = makeNew(dataset)
    for i in range(VOCAB_TRAINING):  
        print("Training iteration " + str(i))
        nodelist = update(nodelist, dataset)

    for i in range(GRAMMAR_TRAINING):  # Train it without expanding vocabulary
        print("Training iteration " + str(VOCAB_TRAINING+i))
        nodelist = update(nodelist, dataset, updateV=False)

    saveChain(nodelist, 'GatsbyChain')

    print("Generating sentence...")
    print(generateSentence(nodelist, LEN_OUTPUT))

    os.remove('out.txt')    # Remove this line to keep vocabulary file

main()