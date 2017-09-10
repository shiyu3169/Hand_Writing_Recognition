/**
 * Created by yixing on 8/15/16.
 *
 * This class is the main class
 * It run the ecoc class and read the model, train prediction,
 * test prediction result from file system that saved by the ecoc program.
 */
public class Main {

    private static double[][] trainFunctionResult;
    private static double[][] testFunctionResult;
    private static double[] trainOrigLabel;
    private static double[] testOrigLabel;
    private static double[][] ecocFunction;

    /**
     * Run the main function to start the program
     * @param args
     */
    public static void main(String[] args) {
        ECOC ecoc = new ECOC();
        ecoc.runECOC();
        readModel();
        System.out.println("The train set is" + trainOrigLabel.length);
        System.out.println("The test set is" + testOrigLabel.length);
        System.out.println("The train set is" + trainFunctionResult.length);
        System.out.println("The test set is" + testFunctionResult.length);
        double trainAcc = predict(trainFunctionResult, trainOrigLabel);
        double testAcc = predict(testFunctionResult, testOrigLabel);
        System.out.println("The train accurancy is: " + trainAcc);
        System.out.println("The test accurancy is: " + testAcc);
    }

    /**
     * read result including ecoc functions, train predict ecoc function and test predict ecoc function
     */
    public static void readModel() {
        DataInput train = new DataInput("trainFunction.txt");
        DataInput test = new DataInput("testFunction.txt");
        DataInput ecoc = new DataInput("ecocFunction.txt");
        DataInput trainLabel = new DataInput("trainLabel.txt");
        DataInput testLabel = new DataInput("testLable.txt");
        trainFunctionResult = train.getData();
        testFunctionResult = test.getData();
        ecocFunction = ecoc.getData();
        trainOrigLabel = transfer2Dto1D(trainLabel.getData());
        testOrigLabel = transfer2Dto1D(testLabel.getData());

    }

    /**
     * Helper function to transfer 2d array to 1d
     * @param matrix
     * @return 1d array
     */
    public static double[] transfer2Dto1D(double[][] matrix) {
        double[] temp = new double[matrix[0].length];
        for(int i = 0; i < matrix[0].length; i++) {
            temp[i] = matrix[0][i];
        }
        return temp;
    }


    /**
     * calculate the accurancy of dataset
     * @param functionResult the train or test predicted ecoc function
     * @param origLabel the orignial label
     * @return the train or test accurancy
     */
    private static double predict(double[][] functionResult, double[] origLabel) {
        double acc = 0.00000;
        for(int i = 0; i < functionResult.length; i++) {
            double temp = getDistant(functionResult[i]);
            System.out.println("The predict label is " + temp + " The true label is " + origLabel[i]);
            if (temp == origLabel[i]) {
                acc++;
            }
        }
        return acc / functionResult.length;
    }

    /**
     * According the ecoc code book, calculate the diff with predict function
     * use the diff determin this prediction belongs to which class
     * @param list
     * @return class label
     */
    private static double getDistant(double[] list) {
//        printArray(list);
//        System.out.println(list.length);
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
}
