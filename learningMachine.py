from sklearn import linear_model
from sklearn import svm
from sklearn import tree
import operator

feature_map = {}
queryDocLabel = {}
queries = []
queryTrain = []
queryTest = []
QUERY_DOC_LIST = []
train_feature_matrix = []
test_feature_matrix = []
train_label_vector = []
test_label_vector = []
training = []
testing = []
new_dict = {}

STORE_RESULT = '/Users/vishruthkrishnaprasad/Downloads/IR/ASSGN6/result2/'
QREL_FILE = '/Users/vishruthkrishnaprasad/Downloads/IR/ASSGN6/AP_DATA/qrels.adhoc.51-100.AP89.txt'
QUERY_FILE = '/Users/vishruthkrishnaprasad/Downloads/IR/ASSGN6/AP_DATA/query_desc.51-100.short.txt'
PERFORMANCE = '/Users/vishruthkrishnaprasad/Downloads/IR/ASSGN6/performance/'
features = ['tfIDF', 'okapiTF', 'bm25', 'lmJM', 'laplace']


def main():
    createQueryList()
    print "The train query list is:", queryTrain
    print "The test query list is:", queryTest
    dictOfQrel()
    buildfeatureMap()
    featureLabelMatrix()
    mlModels()


def createQueryList():
    f = open(QUERY_FILE, "r")
    for line in f.readlines():
        query_id = line.split(".")[0]
        if query_id not in queries:
            queries.append(query_id)
    queries.pop()
    chooseQueries()
    f.close()


def chooseQueries():
    global queryTrain
    global queryTest

    for i in range(len(queries)):
        if 15 <= i < 20:
            continue
        queryTrain.append(queries[i])
    queryTest = list(set(queries) - set(queryTrain))


def dictOfQrel():
    f = open(QREL_FILE, "r")
    for line in f.readlines():
        query_id, _, doc_id, grade = line.split()
        if query_id in queries:
            queryDocLabel[(query_id, doc_id)] = int(grade)
        if query_id not in new_dict:
            new_dict[query_id] = 999
        else:
            new_dict[query_id] -= 1
    f.close()


def buildfeatureMap():
    temp_dict = {}
    for feature in features:
        path = STORE_RESULT + feature
        f = open(path, "r")
        final_score = dict()
        for line in f.readlines():
            query_id, _, doc_id, _, score, _ = line.split()
            score = float(score)
            final_score[query_id] = score
            if (query_id, doc_id) not in temp_dict:
                temp_dict[(query_id, doc_id)] = {feature: score}
            else:
                temp_dict[(query_id, doc_id)][feature] = score

        for pair in queryDocLabel:
            if pair not in temp_dict:
                temp_dict[pair] = {feature: final_score[pair[0]]}
        f.close()


    for pair in queryDocLabel:
        feature_map[pair] = temp_dict[pair]

    for pair in temp_dict:
        query_id = pair[0]
        if pair not in feature_map and new_dict[query_id] != 0:
            feature_map[pair] = temp_dict[pair]
            new_dict[query_id] -= 1

    print "The feature map length is:", len(feature_map)


def featureLabelMatrix():
    for pairs in sorted(feature_map):
        query_id = pairs[0]
        tmp_feature = feature_map[pairs]
        fvector = []
        if pairs not in queryDocLabel:
            grade = 0
        else:
            grade = queryDocLabel[pairs]
        for feature in features:
            if feature in tmp_feature:
                fvector.append(tmp_feature[feature])
            else:
                fvector.append(0)

        if query_id in queryTrain:
            training.append(pairs)
            train_feature_matrix.append(fvector)
            train_label_vector.append(grade)

        elif query_id in queryTest:
            testing.append(pairs)
            test_feature_matrix.append(fvector)
            test_label_vector.append(grade)


def mlModels():
    linReg()
    decTree()
    sVM()


def linReg():
    clf = linear_model.LinearRegression()
    clf.fit(train_feature_matrix, train_label_vector)
    train_result = clf.predict(train_feature_matrix)
    test_result = clf.predict(test_feature_matrix)
    train_result_map, test_result_map = trainTestResults(train_result, test_result)
    writeFile(train_result_map, "lin_train_4")
    writeFile(test_result_map, "lin_test_4")


def decTree():
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_feature_matrix, train_label_vector)
    train_result = clf.predict(train_feature_matrix)
    test_result = clf.predict(test_feature_matrix)
    train_result_map, test_result_map = trainTestResults(train_result, test_result)
    writeFile(train_result_map, "dec_train_4")
    writeFile(test_result_map, "dec_test_4")

def sVM():
    clf = svm.SVC()
    clf.fit(train_feature_matrix, train_label_vector)
    train_result = clf.predict(train_feature_matrix)
    test_result = clf.predict(test_feature_matrix)
    train_result_map, test_result_map = trainTestResults(train_result, test_result)
    writeFile(train_result_map, "svm_train_4")
    writeFile(test_result_map, "svm_test_4")


def trainTestResults(trained_result, tested_result):
    train_result = {}
    test_result = {}
    for i in range(len(trained_result)):
        pairs = training[i]
        query_id = pairs[0]
        doc_id = pairs[1]
        score = trained_result[i]
        if query_id not in train_result:
            train_result[query_id] = {doc_id: score}
        else:
            train_result[query_id][doc_id] = score

    for i in range(len(tested_result)):
        pairs = testing[i]
        query_id = pairs[0]
        doc_id = pairs[1]
        score = tested_result[i]
        if query_id not in test_result:
            test_result[query_id] = {doc_id: score}
        else:
            test_result[query_id][doc_id] = score
    return train_result, test_result


def writeFile(result_map, filename):
    outpath = PERFORMANCE + filename
    f = open(outpath, "w")
    for query_id in result_map:
        result = result_map[query_id]
        sorted_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
        count = 1
        for element in sorted_result:
            if count <= 1000:
                f.write('%s Q0 %s %d %f Exp\n' % (query_id, element[0], count, element[1]))
                count += 1
            else:
                break
    f.close()


if __name__ == '__main__':
    main()
