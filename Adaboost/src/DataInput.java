import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by yixing on 8/11/15.
 * This class used to read file as a 2d double array
 */
public class DataInput {

    protected double[][] data;

    protected String path;

    public DataInput(String path) {

        this.path = path;
        this.data = convertToArray(insertDataToArray(path));
    }



    protected ArrayList<double[]> insertDataToArray (String document) {
        ArrayList<double[]> newData = new ArrayList<double[]>();
        BufferedReader readFile = null;
        try {
            readFile = new BufferedReader(new FileReader(document));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        try {
            while (true) {
                String line = readFile.readLine();
                if (line == null) break;
                if (line.trim() == "") break;
                String[] oneLineStrings = line.trim().split("\\s+|,");
                double[] oneLineDoubles = new double[oneLineStrings.length];
                for (int i = 0; i < oneLineStrings.length; i++) {
                    oneLineDoubles[i] = Double.parseDouble(oneLineStrings[i]);
//                    System.out.println(oneLineDoubles[i]);
                }
                newData.add(oneLineDoubles);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            readFile.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return newData;
    }
    private double[][] convertToArray (ArrayList<double[]> arrayList) {
        int columns = arrayList.get(0).length;
        double[][] matrix = new double[arrayList.size()][columns];
        for (int i = 0; i < arrayList.size(); i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = (arrayList.get(i))[j];
            }
        }
        return matrix;
    }

    public double[][] getData() {
        return data;
    }
}
