import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * Created by yixing on 8/12/16.
 * This class used to run adaboost
 */
public class ECOCStump {
    private ArrayList<HashMap<Integer, Double>> data;
    private int[] labels;

    protected int steps = 200;
    protected double[][] train;
    protected double[][] test;
    private double[] trainLabel;
    private double[] testLabel;

    private double[] distribution;
    private int[] indicator;
    private double[] testError;
    private double[] trainError;

    private Node[] stumps = new Node[steps];

    public ECOCStump(double[][] train, double[][] test, double[] trainLabel, double[] testLabel) {
        this.train = train;
        this.trainLabel = trainLabel;
        this.test = test;
        this.testLabel = testLabel;
        System.out.println("The train label is:");
        printArray(this.trainLabel);
        System.out.println("The test label is:");
        printArray(this.testLabel);
        this.distribution = initDistribution();
        this.indicator = initIndicator();
        this.testError = null;
        this.trainError = null;
    }

    public void adaBoost() {
//        Node[] stumps = new Node[steps];
        for(int i = 0; i < steps; i++) {
            System.out.println("This is " + i + " step");
            Node root = makeStump();
            stumps[i] = root;
            updateDistribution(root);
//            printArray(this.distribution);
//            System.out.println("round test error is " + testErr + ", round tarin error is " + trainErr);
        }
        this.testError = makePredict(stumps, this.test, steps);
        this.trainError = makePredict(stumps, this.train, steps);
    }

    private double[] initDistribution() {
        int length = this.train.length;
        double[] dist = new double[length];
        for(int i = 0; i < length; i++) {
            dist[i] = 1.0 / length;
        }
        return dist;
    }

    private Node makeStump() {
        Node root = makeRoot();
        split(root);
        return root;
    }

    private Node makeRoot() {
        Node root = new Node(indicator, 0.0);
//        System.out.println(indicator.length);
        return root;
    }

    private int[] initIndicator() {
        int length = this.train.length;
        int[] indicator = new int[length];
        for(int i = 0; i < length; i++) {
            indicator[i] = i;
        }
        return indicator;
    }

    protected void split(Node node) {
//        int featureNumber = this.data[0].length-1;
//        System.out.println(featureNumber);
        int featureIndex = 0;
        double threshold = 0.0;
        Node leftnode = null;
        Node rightnode = null;
        double maxError = 0.0;
        double rawError = 0.0;
        double flip = 1.0;
        for(int i = 0; i < this.train[0].length-1; i++) {
            ArrayList<Double> values = new ArrayList<Double>();
            for(int j = 0; j < indicator.length; j++) {
                double featureValue = this.train[indicator[j]][i];
                if(values.contains(featureValue)) {
                    continue;
                } else {
                    values.add(featureValue);
                }
            }
            Collections.sort(values);
//            printArrayList(values);
            for (int k = 0; k<values.size()-1; k++) {
                double currentThreshold = (values.get(k) + values.get(k + 1)) / 2;
//                double currentThreshold = values.get(k);
                ArrayList<Integer> leftIndex = new ArrayList<Integer>();
                ArrayList<Integer> rightIndex = new ArrayList<Integer>();
                for(int j = 0; j < indicator.length; j++) {
                    if(this.train[indicator[j]][i] <= currentThreshold) {
                        leftIndex.add(indicator[j]);
                    } else {
                        rightIndex.add(indicator[j]);
                    }
                }
                double[] leftLabels = indexToLable(leftIndex);
                double[] rightLabels = indexToLable(rightIndex);
                int[] leftIndicator = indexToindicator(leftIndex);
                int[] rightIndicator = indexToindicator(rightIndex);
//                printArray(leftIndicator);
//                printArray(rightIndicator);

                double leftLabel = majorityLabel(leftLabels);
                double rightLabel = majorityLabel(rightLabels);


                double leftError = calculateError(leftLabel, leftIndicator, leftLabels);
                double rightError = calculateError(rightLabel,rightIndicator,rightLabels);

//                System.out.printf("The left error is: %f\n", leftError);
//                System.out.printf("The right error is %f\n", rightError);

                double totalError = leftError + rightError;
                double currentError = Math.abs(0.5 - totalError);
//                System.out.print("Current Error is" + currentError + "\n");
                if(currentError > maxError) {
                    maxError = currentError;
                    rawError = totalError;
//                    System.out.print("Current Error is" + currentError + "\n");
                    featureIndex = i;
//                    if(leftError + rightError > 0.5) {
//                        flip = -1.0;
//                    }
                    threshold = currentThreshold;
                    leftnode = new Node(leftIndicator,leftLabel);
                    rightnode = new Node(rightIndicator, rightLabel);
//                    printArray(leftLabels);
//                    printArray(rightLabels);
//                    System.out.print("The left label is " + leftLabel);
//                    System.out.print("The right label is " + rightLabel);
                }
            }
        }
        if(rawError > 0.5) {
            flip = -1.0;
//            rawError = 1-rawError;
        }
        node.setLeftchild(leftnode);
        node.setRightchild(rightnode);
        node.setFeature(featureIndex);
        node.setThreshold(threshold);
        node.setError(getAlpha(rawError));
    }

    private void updateDistribution(Node root) {
        double alpha = root.getError();
//        System.out.println("alpha is " + alpha);
        double[] newDistribution = new double[root.getIndicator().length];
//        printArray(root.getIndicator());
        int[] leftIndicator = root.getLeftchild().getIndicator();
        int[] rightIndicator = root.getRightchild().getIndicator();
//        printArray(leftIndicator);
//        printArray(rightIndicator);
        double leftLabel = root.getLeftchild().getLabel();
        double rightLabel = root.getRightchild().getLabel();
//        System.out.print("The left node label is" + leftLabel);
//        System.out.print("The right node label is" + rightLabel);

        for(int i = 0; i < leftIndicator.length; i++) {
            newDistribution[leftIndicator[i]] = Math.exp(-alpha * leftLabel * this.trainLabel[leftIndicator[i]]) *
                    this.distribution[leftIndicator[i]];
        }
        for(int i = 0; i < rightIndicator.length; i++) {
            newDistribution[rightIndicator[i]] = Math.exp(-alpha * rightLabel * this.trainLabel[rightIndicator[i]]) *
                    this.distribution[rightIndicator[i]];
        }
        double sum = 0;
        for(int i = 0; i < this.trainLabel.length; i++) {
            sum +=newDistribution[i];
        }
        for(int i = 0; i < this.trainLabel.length; i++) {
            newDistribution[i] = newDistribution[i] / sum;
        }
        this.distribution = newDistribution;
//        printArray(this.distribution);
//        System.out.print("The distribution length is" + this.distribution.length);
    }

    protected double getAlpha(double error) {
        return 0.5 * Math.log((1-error)/error);
    }

    protected double calculateError(double label, int[] indicator, double[] labels){
        double error = 0.0;
        for(int i = 0; i < indicator.length; i++) {
            if(labels[i] != label) {
                error += this.distribution[indicator[i]];
            }
        }
        return error;
    }

    protected double majorityLabel(double[] labels) {
        int neg = 0;
        int pos = 0;
        for(int i = 0; i < labels.length; i++) {
            if(labels[i] == -1) {
                neg++;
            } else {
                pos++;
            }
        }
        return pos > neg? 1.0 : -1.0;
    }

    protected double[] indexToLable(ArrayList<Integer> list) {
        double[] sublabels = new double[list.size()];
        for (int i = 0; i< list.size(); i++) {
            sublabels[i] = this.trainLabel[list.get(i)];
        }
//        printArray(sublabels);
        return sublabels;
    }

    protected int[] indexToindicator(ArrayList<Integer> list) {
        int[] indicator = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            indicator[i] = list.get(i);
        }
        return indicator;
    }

    private double predict(Node node, double[] data) {
        int feature = node.getFeature();
        double threshold = node.getThreshold();
        double alpha = node.getError();
        double res;
//        System.out.println(feature + "\t" + threshold + "\t" + alpha + "\t" + flip);
//        if(!data.containsKey(feature) || data.get(feature) <= threshold) {
//            res = alpha * node.getLeftchild().getLabel();
//        } else {
//            res = alpha * node.getRightchild().getLabel();
//        }
        if(data[feature] <= threshold) {
            res = alpha * node.getLeftchild().getLabel();
        } else {
            res = alpha * node.getRightchild().getLabel();
        }
//        System.out.println(res);
//        System.out.println(res*flip);
        return res;
    }

    private double[] makePredict(Node[] stumps, double[][] data, int length) {
        double[] error = new double[data.length];
//        int labelIndex = this.test[0].length - 1;
        for(int i = 0; i < data.length; i++) {
            double temp = 0.0;
            for(int j = 0; j < length; j++) {
                temp += predict(stumps[j], data[i]);
            }
//            System.out.println("The temp is " + temp + "The test label is:" + this.testLables[i]);
            double label = 1.0;
            if(temp < 0) {label = -1.0;}
//            System.out.println();
            error[i] = label;
        }
        return error;
    }

    private void printArray(double[] list) {
        for (int i = 0; i < list.length; i++) {
            System.out.print(list[i]+ "\t");
        }
        System.out.print("\n");
    }

    public double[] getTestError() {
        return testError;
    }

    public double[] getTrainError() {
        return trainError;
    }
}
