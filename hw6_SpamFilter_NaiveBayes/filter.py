featureSize = 100
stop_words_list = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
                       "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
                       'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
                       'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't",
                       'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
                       "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm",
                       "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more',
                       'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
                       'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she',
                       "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that',
                       "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these',
                       'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too',
                       'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were',
                       "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who',
                       "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll",
                       "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']

import math

def trainDataLoad(location):
    file = open(location)
    fileData = file.read()
    docs = fileData.split("\n")
    spamDocs = {}
    hamDocs = {}
    cs = 0
    ch = 0
    for doc in docs:
        words = doc.split(" ")
        if len(words)==1:
            continue
        emailId = words[0]
        clas = words[1]
        rem = words[2:]
        useDoc = None
        if clas == "spam":
            cs += 1
            useDoc = spamDocs
        else:
            ch += 1
            useDoc = hamDocs
        for i in range(0,len(rem),2):
            try:
                useDoc[rem[i]] += int(rem[i + 1])
            except:
                useDoc[rem[i]] = int(rem[i + 1])

    return ((spamDocs, cs), (hamDocs,ch))


def testdataLoad(location):
    file = open(location)
    fileData = file.read()
    docs = fileData.split("\n")
    data = {}
    actualResult = {}

    for doc in docs:
        words = doc.split(" ")
        if len(words)==1:
            continue
        emailId = words[0]
        clas = words[1]
        rem = words[2:]
        if clas == "spam":
            actualResult[emailId] = True
        else:
            actualResult[emailId] = False

        useDoc = {}
        for i in range(0,len(rem),2):
            try:
                useDoc[rem[i]] += int(rem[i + 1])
            except:
                useDoc[rem[i]] = int(rem[i + 1])

        data[emailId] = useDoc

    return (data, actualResult)


def extractFeatures(docs):
    
    stop_words = {}
    for word in stop_words_list:
        stop_words[word] = 1

    features = {}
    for word, count in docs.items():
        try:
            if stop_words[word] == 1:
                pass
        except:
            features[word] = count

    import operator
    features = sorted(features.items(), key=operator.itemgetter(1), reverse=True)

    features = features[:featureSize]
    total = 0
    print(features)
    print(len(features))

    featuresDict = {}

    for feature, count in features:
        featuresDict[feature] = count
        total += count

    for feature, count in featuresDict.items():
        import math
        featuresDict[feature] = math.log(count / total)

    return featuresDict


def findAccuracy(actual, pred):
    correctRes = 0
    for word, cls in actual.items():
        if cls == pred[word]:
            correctRes+=1

    return len(actual)/correctRes

trainData = trainDataLoad("data/train")
spamDocs, sc = trainData[0]
hamDocs, hc = trainData[1]

ps = math.log(sc/(sc+hc))
ph = math.log(hc/(sc+hc))

spamFeatures = extractFeatures(spamDocs)
hamFeatures = extractFeatures(hamDocs)

testData = testdataLoad("data/test")
testset = testData[0]
actualRes = testData[1]
predRest = {}

for (emailID, words) in testset.items():
