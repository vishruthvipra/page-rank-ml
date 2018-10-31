TESTS = '/Users/vishruthkrishnaprasad/Downloads/IR/ASSGN6/performance/TESTS/'
TRAINS = '/Users/vishruthkrishnaprasad/Downloads/IR/ASSGN6/performance/TRAINS/'
AVG = '/Users/vishruthkrishnaprasad/Downloads/IR/ASSGN6/performance/AVG/'
result_map = dict()


def main():
    global result_map

    printing(TRAINS, "lin_train_")
    printing(TRAINS, "dec_train_")
    printing(TRAINS, "svm_train_")
    printing(TESTS, "lin_test_")
    printing(TESTS, "dec_test_")
    printing(TESTS, "svm_test_")


def printing(folder, ext):
    for i in range(1, 6):
        f = open(folder + ext + str(i), "r")
        for line in f.readlines():
            query_id, _, doc_id, _, score, _ = line.split()
            pair = (query_id, doc_id)
            if pair not in result_map:
                result_map[pair] = float(score)
            else:
                result_map[pair] += float(score)
        f.close()
    tmplist = []
    for pair in result_map:
        result_map[pair] /= 5
        tmplist.append([pair[0], result_map[pair], pair[1]])

    f = open(AVG + ext + "_avg", "w")
    count = 1

    for item in sorted(tmplist, reverse=True):
        if count > 1000:
            count = 1
        pair1 = item[0]
        pair2 = item[2]
        f.write('%s Q0 %s %d %f Exp\n' % (pair1, pair2, count, item[1]))
        count += 1


if __name__ == '__main__':
    main()
