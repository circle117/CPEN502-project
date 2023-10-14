package ece.cpen502;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class BackPropagationMain {

    public static void main(String[] args) throws IOException {
        String xorFilePath = "./data/xor.csv";
        String bipolarFilePath = "./data/bipolar.csv";
        int argA, argB = 1;

        // input the dataset name
        System.out.print("Please input the dataset name (xor/bipolar) : ");
        Scanner scanner = new Scanner(System.in);
        String inputFileName = scanner.next();

        File dataset;
        if (inputFileName.equals("xor")) {
            argA = 0;
            dataset = new File(xorFilePath);
        }
        else if (inputFileName.equals("bipolar")) {
            argA = -1;
            dataset = new File(bipolarFilePath);
        }
        else {
            System.out.println("You give the wrong dataset name.");
            return;
        }

        // input the momentum vale
        System.out.print("Please input the momentum value: ");
        double momentum = Double.parseDouble(scanner.next());

        // input whether record the weights or not
        System.out.print("Do you want to record the weights in an csv file? (Y/N) ");
        String ifRecord = scanner.next();

        System.out.print("How many times do you want to run? ");
        int times = Integer.parseInt(scanner.next());

        // load the dataset
        Scanner fileScanner = new Scanner(dataset);
        fileScanner.next();

        double[][] X = new double[4][2];
        double[] target = new double[4];
        int cnt = 0;
        while (fileScanner.hasNext()) {
            String[] str = fileScanner.next().split(",");
            X[cnt][0] = Integer.parseInt(str[0]);
            X[cnt][1] = Integer.parseInt(str[1]);
            target[cnt] = Integer.parseInt(str[2]);
            cnt++;
        }

        FileWriter writer;

        // train
        cnt = 0;
        double error;
        int epoch;
        double totalEpochs = 0;
        ArrayList<List<Double>> totalList = new ArrayList<>();
        while (cnt<times) {
            List<Double> list = new ArrayList<>();

            NeuralNet net = new NeuralNet(2, 4, 0.2, momentum,
                    argA, argB);

            error = 1.0;
            epoch = 0;
            while (error >= 0.05) {
                error = 0.0;
                for (int i = 0; i < X.length; i++) {
                    error += net.train(X[i], target[i]);
                }
                epoch++;
                if (ifRecord.equals("Y"))
                    list.add(error);
                    // writer.write(epoch+","+error+"\n");
            }

            if (ifRecord.equals("Y"))
                totalList.add(list);
            totalEpochs += epoch;
            cnt++;
            System.out.printf("Epoch: %d, Error: %.8f\n", epoch, error);
        }
        System.out.printf("The average number of converging epoch is %d\n", Math.round(totalEpochs/cnt));

        // record
        if (ifRecord.equals("Y")) {
            File file = new File("./data/result/"+inputFileName+"Loss-momentum"+momentum+".csv");
            if (file.exists())
                file.delete();
            writer = new FileWriter(file);
            writer.write("epoch");
            for (int i=1; i<=10; i++)
                writer.write(",loss"+i);
            writer.write("\n");

            int j = 0;
            while (true) {
                writer.write(String.valueOf(j));
                cnt = 0;
                for (int i = 0; i < 10; i++) {
                    if (totalList.get(i).size()-1<j) {
                        writer.write(",");
                        cnt++;
                    }
                    else
                        writer.write(","+totalList.get(i).get(j));
                }
                if (cnt==times)
                    break;
                writer.write("\n");
                j++;
            }
            writer.close();
        }
    }
}
