package ece.cpen502;


import ece.cpen502.robot.State;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class OfflineTraining {
    private final String LUTFileName = "./data/data-e0.9-round4000-alpha0.5-gamma0.99-LUT.csv";
    private final static String NNFileName = "./data/assignment-3/offline-data-hidden%d-learningRate%f-momentum%.1f.csv";
    private final static String recordFileName = "./data/assignment-3/best-result-10.csv";
    private final static int argNumHidden = 10;
    private final static double learningRate = 1.2;
    private final static double momentum = 0.9;
    private final static State state = new State();
    private final static double QVALUE_MAX = 0.24979481038987;
    private final static double QVALUE_MIN = 0;

    public ArrayList<double[]> loadLUT() throws FileNotFoundException {
        ArrayList<double []> data = new ArrayList<>();

        System.out.println("load from "+ LUTFileName);
        File file = new File(LUTFileName);
        Scanner scanner = new Scanner(file);
        scanner.next();
        while (scanner.hasNext()) {
            // state 1, state 2, state 3, state 4, Q(S, A1), Q(S, A2), Q(S, A3), Q(S, A4), Q(S, A5)
            String[] line  = scanner.next().split(",");
            double[] row = new double[9];
            row[0] = Double.parseDouble(line[0]);
            row[1] = Double.parseDouble(line[1]);
            row[2] = Double.parseDouble(line[2]);
            row[3] = Double.parseDouble(line[3]);
            row[4] = (Double.parseDouble(line[5]) - QVALUE_MIN)/(QVALUE_MAX - QVALUE_MIN);
            for (int i=5; i<9; i++) {
                String[] subline  = scanner.next().split(",");
                row[i] = (Double.parseDouble(subline[5]) - QVALUE_MIN)/(QVALUE_MAX - QVALUE_MIN);
            }
            data.add(row);
        }
        return data;
    }

    public static void main(String [] args) throws IOException {
        OfflineTraining offline = new OfflineTraining();

        ArrayList<double[]> data = offline.loadLUT();

        // File file = new File(recordFileName);
        // FileWriter writer = new FileWriter(file);
        // writer.write("epoch,rms\n");
        // String[] momentumString = {"epoch", "momentum0", "momentum0.1", "momentum0.2", "momentum0.3", "momentum0.4",
        //         "momentum0.5", "momentum0.6", "momentum0.7", "momentum0.8", "momentum0.9\n"};
        // double[] momentums = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        // writer.write(String.join(",", momentumString));
        // ArrayList<List<Double>> errorList = new ArrayList<>();

        double error = 0.5;
        double rms = 0;
        int cnt;
        int epoch = 1;
        double[] errors;

        NeuralNet net = new NeuralNet(4, argNumHidden, 5,
                learningRate, momentum, 0, 1, true);
        while (error >= 0.002) {
            cnt = 0;
            error = 0;
            while (cnt < data.size()) {
                double[] row = data.get(cnt);
                double[] preprocessRow = new double[4];
                preprocessRow[0] = state.EnergyMinMaxScaling(state.EnergyTransferBack(row[0]));
                preprocessRow[1] = state.DistanceMinMaxScaling(state.DistanceTransferBack(row[1]));
                preprocessRow[2] = state.EnergyMinMaxScaling(state.EnergyTransferBack(row[2]));
                preprocessRow[3] = state.DistanceMinMaxScaling(state.DistanceTransferBack(row[3]));
                errors = net.train(Arrays.copyOf(preprocessRow, 4), Arrays.copyOfRange(row, 4, 9));
                for (double e : errors) {
                    error += e;
                    rms += 2 * e;
                }
                cnt += 5;
            }
            error /= cnt;
            rms = Math.sqrt(rms / cnt);
            if (epoch % 100 == 0) {
                System.out.printf("Epoch: %d, Error: %.8f, RMS: %.8f\n", epoch, error, rms);
            }
            epoch++;
        }
        // writer.write(epoch+","+rms+"\n");
        // writer.close();
        // System.out.printf("Epoch: %d, Error: %.8f, RMS: %.8f\n", epoch, error, rms);

        File file = new File(String.format(NNFileName, argNumHidden, learningRate, momentum));
        net.save(file);
    }
}
