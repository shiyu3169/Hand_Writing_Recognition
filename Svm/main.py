import scipy.io
import sys
import svm as svm_module
import scipy as s

def evaluate(svm, datapoints, classes):
    size = len(datapoints)
    output_classes = svm.classify_2d(datapoints)
    diff_classes = classes - output_classes
    errors = s.count_nonzero(diff_classes)
    classified_correctly = size - errors
    print "Correct: %u/%u, %.2f%%" % (classified_correctly, size,
                                      100.0 * classified_correctly/size)

DEFAULT_FILENAME = 'MNIST.mat'


def train_and_test(filename = ''):
    if filename == '':
        filename = DEFAULT_FILENAME

    d = scipy.io.loadmat(filename)

    train_datapoints = d['Xtrain']
    train_classes = d['Ytrain'].flatten()
    test_datapoints = d['Xtest']
    test_classes = d['Ytest'].flatten()

    dSize = 1000
    print 'Dataset size:', dSize

    train_datapoints = train_datapoints[:dSize]
    train_classes = train_classes[:dSize]

    dataset_size = len(train_datapoints)
    dim = len(train_datapoints[0])

    svm = svm_module.SVM(train_datapoints[:dSize], train_classes[:dSize])
    svm.set_params(C=3.2, tau=0.008)
    svm.run()

    print "Evaluating on a train dataset..."
    evaluate(svm, train_datapoints, train_classes)
    print "Evaluating on a test dataset..."
    evaluate(svm, test_datapoints, test_classes)

if __name__ == '__main__':
    if len(sys.argv) not in [1,2]:
        print >> sys.stderr
        sys.exit(1)

    if len(sys.argv) == 1:
        sys.argv.append('')
    train_and_test(sys.argv[1])

