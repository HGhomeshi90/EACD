import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Main {

    public static final int Feature_Percent = 70;
    public static final int Subspaces_Number = 7;
    public static  final int StartRemovingClassifiersIteration = 4;
    public static final int Data_Block_Instances = 1000;
    public static final int Data_Block_Max_Classifier = 20;
    public static final int Data_Block_Min_Classifier = 1;
    
    private static List<DataBlock> ss = new ArrayList<>();
    private static int iterationNo = 1;

    public static void main(String[] args) throws IOException {


        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader("elecNormNew.arff"));
            Instances data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);

            int start = 0;

            Instances roundData = new Instances(data, start, Data_Block_Instances);

            int featureCount = data.numAttributes() * Feature_Percent / 100;

            for (int i = 0; i < Subspaces_Number; i++) {
                ss.add(new DataBlock(roundData, featureCount));
            }
            int round = 1;

            do {
//                System.out.println("Round " + round);
                start += Data_Block_Instances;
                Instances round2Data = new Instances(data, start, Data_Block_Instances);

                System.out.println(getAlgorithmAccuracy(round2Data));
//                System.out.println("*****************************");

                double avgAccuracy = getAvgAccuracy(round2Data);

//                System.out.println("*****************************");

                updateDataBlockClassifiers(round2Data, avgAccuracy);


//                System.out.println("--------------------------------------------------------------------");
                round++;


            } while (start < data.numInstances());


        } catch (Throwable e) {
            e.printStackTrace();
        } finally {
            if (reader != null)
                reader.close();
        }
    }

    private static double getAlgorithmAccuracy(Instances round2Data) throws Exception {
        double accuracy = 0;
        List<List<Double>> classes = new ArrayList<>();
        for (int i = 0; i < round2Data.numInstances(); i++) {
            classes.add(new ArrayList<>());
        }
        for (DataBlock cw : ss) {
            List<List<Double>> c = cw.getInctancesClass(round2Data);
            for (int i = 0; i < c.size(); i++) {
                classes.get(i).addAll(c.get(i));
            }
        }

        for (int i = 0; i < round2Data.numInstances(); i++) {
            if (getMode(classes.get(i)) == round2Data.get(i).classValue())
                accuracy++;
        }

        return accuracy / round2Data.numInstances();
    }

    private static double getMode(List<Double> arr) {
        int modeCount = 0;
        double mode = 0;

        int currCount = 0;
        double currElement;

        for (double candidateMode : arr)
        {
            currCount = 0;

            for (double element : arr)
            {
                if (candidateMode == element)
                {
                    currCount++;
                }
            }
            if (currCount > modeCount)
            {
                modeCount = currCount;
                mode = candidateMode;
            }
        }

        return mode;
    }

    private static void updateDataBlockClassifiers(Instances round2Data, double avgAccuracy) throws Exception {
        for (int i = 0; i < Subspaces_Number; i++) {
            DataBlock db = ss.get(i);
            if (db.getDataBlockAccuracy() >= avgAccuracy) {
                db.updateInstances(round2Data);
            }
            else if(iterationNo  >= StartRemovingClassifiersIteration){
                db.removeWorstClassier();
            }
//            System.out.println("Classifiers in Data Block " + (i + 1) + ":" + db.getClassifiers().size());
        }
        iterationNo++;
        
        
    }


    private static double getAvgAccuracy(Instances round2Data) throws Exception {
        double avgAccuracy = 0;

        for (int i = 0; i < Subspaces_Number; i++) {
            double dbAccuracy = ss.get(i).getBlockAccuracy(round2Data);
            avgAccuracy += dbAccuracy;
//            System.out.println("Data Black " + (i + 1) + " Accuracy:" + dbAccuracy);
        }

        avgAccuracy = avgAccuracy / Subspaces_Number;
        return avgAccuracy;
    }
}
