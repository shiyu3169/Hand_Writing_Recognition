import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Created by yixing on 8/11/16.
 * This class is for HAAR extraction.
 */
public class HAARFeatureExtraction {

    private int rectangleNum = 100;
    private int numRow;
    private int numCol;
    private int[][] rectangles;
    private int coord = 4;
    private double[][] dataFeatures;


    public HAARFeatureExtraction() {
        this.numRow = 0;
        this.numCol = 0;
        this.dataFeatures = null;
        this.rectangles = null;
    }


    public double[][] run(ArrayList<int[][]> images) {
        // read image list, generate random rectangle coordinate list, run HAAR on all images
        dataFeatures = new double[images.size()][rectangles.length * 2];
        for(int i = 0; i < images.size(); i++) {
            System.out.print("This is the " + i + "step");
            double[] instance = runHAAR(images.get(i));
            print1DArray(instance);
            dataFeatures[i] = instance;
        }
//        writeToFile(dataFeatures);
        return dataFeatures;
    }

    public void initialize(ArrayList<int[][]> images) {
        // initialze numRow, numCol
        numRow = images.get(0).length;
        numCol = images.get(0)[0].length;
        rectangles = generateRandomRectangle();
    }

    private double[] runHAAR(int[][] image) {
        int[][] dp = generateDP(image);
        ArrayList<Double> features = new ArrayList<Double>();
        for(int i = 0; i < rectangles.length; i++) {
            double[] feature = getFeatureValue(dp, rectangles[i]);
            for(int j = 0; j < feature.length; j++) {
                features.add(feature[0]);
            }
        }
        double[] featuresValue = new double[features.size()];
        for(int i = 0; i < features.size(); i++) {
            featuresValue[i] = features.get(i);
        }
        return featuresValue;
    }

    private double[] getFeatureValue(int[][] dp, int[] coordinates) {
        int Ax = coordinates[0];
        int Ay = coordinates[2];
        int Dx = coordinates[1];
        int Dy = coordinates[3];
        int[] A = new int[]{Ax, Ay};
        int[] B = new int[]{Ax, Dy};
        int[] C = new int[]{Dx, Ay};
        int[] D = new int[]{Dx, Dy};
        int[] Q = new int[]{(Ax + Dx)/2, Ay};
        int[] R = new int[]{(Ax + Dx)/2, Dy};
        int[] M = new int[]{Ax, (Ay + Dy)/2};
        int[] N = new int[]{Dx, (Ay + Dy)/2};
        double[] features = new double[2];
        features[0] = getBlack(dp, A, B, Q, R) - getBlack(dp, Q, R, C, D);
        features[1] = getBlack(dp, A, M, C, N) - getBlack(dp, M, C, N, D);
        return features;
    }

    private int[][] generateRandomRectangle() {
        int[][] areas = new int[rectangleNum][coord];
        for(int i = 0; i < rectangleNum; i++) {
            // the first and second represent horizontal, the thrid and forth represent vertical
            ArrayList<Integer> horizontalPoint = generateOneDirection(numRow);
            ArrayList<Integer> verticalPoint = generateOneDirection(numCol);
            int[] list = combineCoordinates(horizontalPoint,verticalPoint);
            areas[i] = list;
        }

        return areas;
    }

    private int[][] generateDP(int[][] image) {
        int[][] dp = new int[numRow][numCol];
        dp[0][0] = image[0][0];
        for(int i = 1; i < numCol; i++) {
            dp[0][i] = image[0][i] + dp[0][i-1];
        }
        for(int j = 1; j < numRow; j++) {
            dp[j][0] = image[j][0] + dp[j-1][0];
        }
        for(int i = 1; i < numRow; i++) {
            for(int j = 1; j < numCol; j++) {
                dp[i][j] = dp[i][j-1] + dp[i-1][j] - dp[i-1][j-1] + image[i][j];
            }
        }
//        print2DArray(dp);
        return dp;
    }

    private double getBlack(int[][] dp, int[] A, int[] B, int[] C, int[] D) {
        double featureValue = dp[D[0]][D[1]] - dp[B[0]][B[1]] - dp[C[0]][C[1]] + dp[A[0]][A[1]];
        return featureValue;
    }

    private void print2DArray(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        for(int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                System.out.print(matrix[i][j] + "\t");
            }
            System.out.print("\n");
        }
    }

    private ArrayList<Integer> generateOneDirection(int range) {
        ArrayList<Integer> cords = new ArrayList<Integer>();
        Random rand = new Random();
        while(cords.size() != 2) {
            int temp = rand.nextInt(range);
            cords.add(temp);
        }
        System.out.print(cords.size());
        Collections.sort(cords);
        return cords;
    }

    private int[] combineCoordinates(ArrayList<Integer> horizontal, ArrayList<Integer> vertical) {
        int size = horizontal.size() + vertical.size();
        int[] array = new int[size];
        for(int i = 0; i < horizontal.size(); i++) {
            array[i] = horizontal.get(i);
        }
        for(int j = 0; j < vertical.size(); j++) {
            array[horizontal.size() + j] = vertical.get(j);
        }
        return array;
    }


    private void print1DArray(double[] array) {
        for(int i = 0; i < array.length; i++) {
            System.out.print(array[i] + "\t");
        }
        System.out.print("\n");
    }


}
