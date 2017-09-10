import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

/**
 * Created by yixing on 8/11/15.
 * This class is to extract features from image data
 * Read image data by MNISTReader class and run HAAR to extract the 200 features from every image file
 * And write to new files with new features
 */
public class ImageFeatureExtraction {

    public static ArrayList<int[][]> trainImages;
    public static ArrayList<int[][]> testImages;
    public static String trainData = "train-images.idx3-ubyte";
    public static String trainLabel = "train-labels.idx1-ubyte";
    public static String testData = "t10k-images.idx3-ubyte";
    public static String testLabel = "t10k-labels.idx1-ubyte";
    public static ArrayList<Double> trainLabelList;
    public static ArrayList<Double> testLabelList;
    public static String trainFile = "trainImage.txt";
    public static String testFile = "testImage.txt";

    public static void main(String[] args) {
        MNISTReader mnistReader = new MNISTReader();
        try {

            mnistReader.reader(trainLabel, trainData);
            trainImages = mnistReader.getImageArrayList();
            trainLabelList = mnistReader.getLabelArrayList();
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
//            MNISTReader mnistReader = new MNISTReader();
            mnistReader.reader(testLabel, testData);
            testImages = mnistReader.getImageArrayList();
            testLabelList = mnistReader.getLabelArrayList();
        } catch (IOException e) {
            e.printStackTrace();
        }

        HAARFeatureExtraction haar = new HAARFeatureExtraction();
        haar.initialize(trainImages);
        double[][] train = haar.run(trainImages);
        double[][] test = haar.run(testImages);
        writeToFile(train, trainLabelList, trainFile);
        writeToFile(test, testLabelList, testFile);

    }

    public static void writeToFile(double[][] matrix, ArrayList<Double> label, String filePath) {
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
            writer.print(label.get(i));
            writer.print("\n");
        }
        writer.close();
    }
}
