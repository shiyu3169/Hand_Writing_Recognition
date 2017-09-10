import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by yixing on 8/12/16.
 * This class construct the ecoc object.
 * In this object, it initialize a ecoc code book and for each function run the ada boost
 * After building the model and prediction, it will write the result to file for reload convenience.
 * Also, it provide function to calculate the accuracy
 */
public class ECOC {

    private DataInput trainData = new DataInput("trainImage.txt");
    private DataInput testData = new DataInput("testImage.txt");

    private double[] trainOrigLabel;
    private double[] testOrigLabel;
    private double[][] trainingOrig;
    private double[][] testingOrig;
    private double[][] training;
    private double[][] testing;

    private int classNumber = 10;
    private int functionNumber = 50;

    private double[][] trainFunctionResult;
    private double[][] testFunctionResult;

    private double[][] ecocFunction;

    /**
     * Construct function, choose train and test size, get dataset label and initialize
     * the ecoc code book
     */
    public ECOC() {
        this.trainingOrig = trainData.getData();
        this.testingOrig = testData.getData();
        this.training = Arrays.copyOfRange(this.trainingOrig, 5000, 5500);
        this.testing = Arrays.copyOfRange(this.testingOrig, 1000, 1200);
        this.trainOrigLabel = getLabels(training);
        this.testOrigLabel = getLabels(testing);
        printArray(this.trainOrigLabel);
        printArray(this.testOrigLabel);
        this.ecocFunction = getEcocFunction();
        this.classNumber = this.ecocFunction.length;
        this.functionNumber = this.ecocFunction[0].length;
        this.trainFunctionResult = new double[this.trainOrigLabel.length][this.functionNumber];
        this.testFunctionResult = new double[this.testOrigLabel.length][this.functionNumber];
    }

    /**
     * main function, it could run the ecoc independently
     * @param args
     */
    public static void main(String[] args) {
        ECOC ecoc = new ECOC();
        ecoc.runECOC();
    }

    /**
     * For each function, change the train and test label to ecoc code,
     * run Adaboost to build model, predict train and test
     * write the result to file system
     * calculate the accuracy
     */
    public void runECOC() {
        for(int i = 0; i < this.functionNumber; i++) {
            System.out.println("This is the " + i + "function step");
            double[] trainLabel = changeLabel(i, this.trainOrigLabel);
            double[] testLabel = changeLabel(i, this.testOrigLabel);
            runAdaBoost(i, trainLabel, testLabel);
        }
        write2DArrayToFile(this.trainFunctionResult, "trainFunction.txt");
        write2DArrayToFile(this.testFunctionResult, "testFunction.txt");
        write1DArrayToFile(this.trainOrigLabel, "trainLabel.txt");
        write1DArrayToFile(this.testOrigLabel, "testLable.txt");
        double trainAcc = predict(this.trainFunctionResult, this.trainOrigLabel);
        double testAcc = predict(this.testFunctionResult, this.testOrigLabel);
        System.out.println("The train accurancy is: " + testAcc);
        System.out.println("The test accurancy is: " + trainAcc);
    }

    /**
     * Initialize a adaboost object with train and test dataset and label
     * according the adaboost result, update the train and test ecoc prediction
     * @param functionNumber ecoc code book function index
     * @param trainLabel changed train label
     * @param testLabel changed test label
     */
    private void runAdaBoost(int functionNumber, double[] trainLabel, double[] testLabel){
        ECOCStump stump = new ECOCStump(this.training, this.testing, trainLabel, testLabel);
        stump.adaBoost();
        double[] trainResult = stump.getTrainError();
        double[] testResult = stump.getTestError();
        addResult(functionNumber, this.trainFunctionResult, trainResult);
        addResult(functionNumber, this.testFunctionResult, testResult);
    }

    private double[][] getEcocFunction() {

        double[][] res = new double[classNumber][functionNumber];
        ArrayList<double[]> temp = new ArrayList<double[]>();
        for(int i = 0; i < functionNumber; i++) {
            double[] tempFunction = getOneEcocFunction();
            while(temp.contains(tempFunction)) {
                tempFunction = getOneEcocFunction();
            }
            for(int j = 0; j < classNumber; j++) {
                res[j][i] = tempFunction[j];
            }
        }
        print2DArray(res);
//        write2DArrayToFile(res, "ecocFunction.txt");
        return res;
    }

    private double[] getOneEcocFunction() {
        Random rand = new Random();
        int bianary = rand.nextInt();
        String bi = Integer.toBinaryString(bianary);
//        System.out.print(bi);
        double[] array = new double[classNumber];
        for(int k = 0; k < classNumber; k++) {
            array[k] = Double.parseDouble(Character.toString(bi.charAt(k)));
            if(array[k] == 0.0) {
                array[k] = -1.0;
            }
        }
        System.out.print("The class fucntion is:");
        printArray(array);
        return array;
    }


    private double[] changeLabel(int functionNumber, double[] origLabel) {
        double[] function = new double[classNumber];
        for(int i = 0; i < classNumber; i++) {
            function[i] = this.ecocFunction[i][functionNumber];
        }
        double[] newLabels = new double[origLabel.length];
        for(int i = 0; i < origLabel.length; i++) {
            int classIndex = (int)origLabel[i];
            double tempLabel = function[classIndex];
            newLabels[i] = tempLabel;
        }
        return newLabels;
    }

    private void addResult(int functionNumber, double[][] functionResult, double[] result) {
        for(int i = 0; i < functionResult.length; i++) {
            functionResult[i][functionNumber] = result[i];
        }
    }

    private double predict(double[][] functionResult, double[] origLabel) {
        double acc = 0.0000;
        for(int i = 0; i < functionResult.length; i++) {
            double temp = getDistant(functionResult[i]);
            System.out.println("The predict label is " + temp + " The true label is " + origLabel[i]);
            if (temp == origLabel[i]) {
                acc++;
            }
        }
        return acc / functionResult.length;
    }

    private double getDistant(double[] list) {
        double minDistance = list.length+1;
        double classIndex = 0.0;
        for(int i = 0; i < ecocFunction.length; i++) {
            if(list.equals(ecocFunction[i])) {
                return (double)i;
            } else {
                double distance = 0.0;
                for(int j = 0; j < list.length; j++) {
                    if(ecocFunction[i][j] != list[j]) {
                        distance++;
                    }
                }
                if(minDistance > distance) {
                    classIndex = i;
                    minDistance = distance;
                }
            }
        }
        return classIndex;
    }

    private void printArray(double[] list) {
        for (int i = 0; i < list.length; i++) {
            System.out.print(list[i]+ "\t");
        }
        System.out.print("\n");
    }

    private void print2DArray(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            printArray(matrix[i]);
        }
    }

    private double[] getLabels(double[][] matrix) {
        int index = matrix[0].length - 1;
        int length = matrix.length;
        double[] labels = new double[length];
        for(int i = 0; i < length; i++) {
            labels[i] = matrix[i][index];
        }
        return labels;
    }

    private void write2DArrayToFile(double[][] matrix, String filePath) {
        PrintWriter writer = null;
        try {
            writer = new PrintWriter(filePath, "UTF-8");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        for(int i = 0; i < matrix.length; i++) {
            for(int j = 0; j < matrix[0].length; j++) {
                writer.print(matrix[i][j] + "\t");
            }
            writer.print("\n");
        }
        writer.close();
    }

    private void write1DArrayToFile(double[] array, String filePath) {
        PrintWriter writer = null;
        try {
            writer = new PrintWriter(filePath, "UTF-8");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        for(int i = 0; i < array.length; i++) {
            writer.print(array[i] + "\t");
        }
        writer.print("\n");
        writer.close();
    }

    public double[] getTrainOrigLabel() {
        return this.trainOrigLabel;
    }

    public double[] getTestOrigLabel() {
        return this.testOrigLabel;
    }
}
