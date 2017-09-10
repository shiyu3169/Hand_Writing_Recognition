/**
 * Created by yixing on 8/12/15.
 * This class is a decision tree class with 1 level.
 * It will be used in adaboost as stump
 */
public class Node {

    private Node leftchild;
    private Node rightchild;
    private int feature;
    private double threshold;
    private int[] indicator;
    private double error;
    private double label;
//    private double flip;

    public Node(int[] indicator, double label) {
        this.indicator = indicator;
        this.threshold = 0.0;
        this.feature = 0;
        this.leftchild = null;
        this.rightchild = null;
        this.error = error;
        this.label = label;
//        this.flip = 1.0;

    }

    private double[] getLabel(int[] indicator, double[][] matrix) {
        double[] label = new double[indicator.length];
        for(int i = 0; i < indicator.length; i++) {
            int index = indicator[i];
            label[i] = matrix[index][matrix[index].length-1];
        }
        return label;
    }

    public Node getLeftchild() {
        return leftchild;
    }

    public void setLeftchild(Node leftchild) {
        this.leftchild = leftchild;
    }

    public Node getRightchild() {
        return rightchild;
    }

    public void setRightchild(Node rightchild) {
        this.rightchild = rightchild;
    }

    public int getFeature() {
        return feature;
    }

    public void setFeature(int feature) {
        this.feature = feature;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    public int[] getIndicator() {
        return indicator;
    }

    public void setIndicator(int[] indicator) {
        this.indicator = indicator;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public double getLabel() {
        return label;
    }

    public void setLabel(double label) {
        this.label = label;
    }

}
