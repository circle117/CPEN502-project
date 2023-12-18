package ece.cpen502;

import ece.cpen502.interfaces.LUTInterface;
import robocode.RobocodeFileWriter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class LookUpTable implements LUTInterface {

    private double [][][][][] lookUpTable;
    private int [][][][][] visits;
    private final int variableDimension1;
    private final int variableDimension2;
    private final int variableDimension3;
    private final int variableDimension4;
    private final int variableDimension5;


    // hyper parameter
    private double epsilon = 0.9;                   // for exploration
    private final double epsilonDecay;
    private final int totalRoundNum = 4000;
    private final double alpha = 0.5;
    private final double gamma = 0.99;

    private double preAction = -1.0;
    private double[] preState = {-1.0, -1.0, -1.0, -1.0};

    // For recording
    private int roundNum = 0;
    private final int recordInterval = 50;
    private int winsPerInterval = 0;
    private ArrayList<Double> rewards = new ArrayList<>();
    private ArrayList<Long> timeEveryRound = new ArrayList<>();
    private File robotFile;
    private final String recordFileName;
    private RobocodeFileWriter writer;


    public LookUpTable(int variableDimension1, int variableDimension2, int variableDimension3,
                       int variableDimension4, int variableDimension5) {
        this.variableDimension1 = variableDimension1;
        this.variableDimension2 = variableDimension2;
        this.variableDimension3 = variableDimension3;
        this.variableDimension4 = variableDimension4;
        this.variableDimension5 = variableDimension5;
        initializeLUT();
        epsilonDecay = epsilon/(totalRoundNum * 0.8);
        recordFileName = String.format("data-e%.1f-round%d-alpha%.1f-gamma%.2f.csv", epsilon, totalRoundNum, alpha, gamma);
    }

    @Override
    public double[] outputFor(double[] X) {
        int actionIndex = 0;
        int[] stateIndex = indexFor(X);
        double maxProbability = Double.MIN_VALUE;
        boolean allZero = true;
        for (int i=0; i<variableDimension5; i++) {
            double curValue = lookUpTable[stateIndex[0]][stateIndex[1]][stateIndex[2]][stateIndex[3]][i];
            if (curValue != 0) allZero = false;
            if (curValue > maxProbability) {
                actionIndex = i;
                maxProbability = curValue;
            }
        }
        if (allZero) actionIndex = (int)Math.floor(Math.random()*variableDimension5);
        return new double[]{actionIndex};
    }

    @Override
    public double[] train(double[] X, double[] argValue) {
        double action;
        double[] maxAction = outputFor(X);

        double random = Math.random();          // [0,1)
        if (random < epsilon)
            action = Math.floor(Math.random() * variableDimension5);
        else
            action = maxAction[0];

        if (preAction >= 0)
            update(X, 0, (int)maxAction[0]);           // on-policy: update(X, 0, (int)action)

        preAction = action;
        preState = Arrays.copyOf(X, X.length);
        int[] stateIndex = indexFor(X);
        visits[stateIndex[0]][stateIndex[1]][stateIndex[2]][stateIndex[3]][(int)action]++;
        return new double[]{action};
    }

    @Override
    public void save(File argFile) throws IOException {
        RobocodeFileWriter writerSave = new RobocodeFileWriter(argFile.getPath() +
                "/"+ recordFileName.substring(0, recordFileName.length()-4) + "-" +"LUT.csv");
        writerSave.write(String.join(",",
                "State-1", "State-2", "State-3", "State-4", "Action", "value", "visitTimes\n"));
        for (int i=0; i< lookUpTable.length; i++)
            for (int j=0; j<lookUpTable[i].length; j++)
                for (int k=0; k<lookUpTable[i][j].length; k++)
                    for (int l=0; l<lookUpTable[i][j][k].length; l++)
                        for (int m=0; m<lookUpTable[i][j][k][l].length; m++)
                            writerSave.write(String.join(",",
                                    String.valueOf(i), String.valueOf(j),
                                    String.valueOf(k), String.valueOf(l), String.valueOf(m),
                                    String.valueOf(lookUpTable[i][j][k][l][m]),
                                    visits[i][j][k][l][m]+"\n"));
        writerSave.close();
        System.out.println("save to "+ argFile.getPath() +
                "/"+ recordFileName.substring(0, recordFileName.length()-4) + "-" +"LUT.csv");
    }

    @Override
    public void load(String argFileName) throws IOException {
        System.out.println("load from "+ argFileName);
        File file = new File(argFileName);
        Scanner scanner = new Scanner(file);
        scanner.next();
        while (scanner.hasNext()) {
            String[] row  = scanner.next().split(",");
            int a = Integer.parseInt(row[0]);
            int b = Integer.parseInt(row[1]);
            int c = Integer.parseInt(row[2]);
            int d = Integer.parseInt(row[3]);
            int e = Integer.parseInt(row[4]);
            lookUpTable[a][b][c][d][e] = Double.parseDouble(row[5]);
            visits[a][b][c][d][e] = Integer.parseInt(row[6]);
        }
    }

    @Override
    public void initializeLUT() {
        lookUpTable = new double[variableDimension1][variableDimension2][variableDimension3][variableDimension4][variableDimension5];
        visits = new int[variableDimension1][variableDimension2][variableDimension3][variableDimension4][variableDimension5];
    }

    @Override
    public int[] indexFor(double[] X) {
        int[] stateIndex = new int[4];
        stateIndex[0] = (int) X[0];
        stateIndex[1] = (int) X[1];
        stateIndex[2] = (int) X[2];
        stateIndex[3] = (int) X[3];
        return stateIndex;
    }

    public void setWriter(File file) throws IOException {
        robotFile = file;
        writer = new RobocodeFileWriter(new File(robotFile.getPath()+"/"+recordFileName));
        writer.write(String.join(",",
                "totalRoundNumber", "winsPer50", "epsilon", "meanTotalReward", "meanTime\n"));
    }

    public void update(double[] X, double reward, int newAction) {
        rewards.add(reward);

        int[] newStateIndex = indexFor(X);
        int[] preStateIndex = indexFor(preState);
        double newQ = lookUpTable[newStateIndex[0]][newStateIndex[1]][newStateIndex[2]][newStateIndex[3]][newAction];
        double oldQ = lookUpTable[preStateIndex[0]][preStateIndex[1]][preStateIndex[2]][preStateIndex[3]][(int)preAction];
        lookUpTable[preStateIndex[0]][preStateIndex[1]][preStateIndex[2]][preStateIndex[3]][(int)preAction] +=
                alpha * (reward + gamma * newQ - oldQ);
    }

    public void terminalState(double reward, long time) throws IOException {
        roundNum++;
        timeEveryRound.add(time);
        rewards.add(reward);
        if (roundNum % recordInterval == 0) {
            double mean = 0;
            for (double element: rewards)
                mean += element;
            mean /= rewards.size();

            long meanTime = 0;
            for (long element: timeEveryRound)
                meanTime += element;
            meanTime /= timeEveryRound.size();
            writer.write(String.join(",",
                    String.valueOf(roundNum), String.valueOf(winsPerInterval*1.0/recordInterval),
                    String.valueOf(epsilon), String.valueOf(mean), meanTime + "\n"));
            rewards = new ArrayList<>();
            winsPerInterval = 0;
        }

        if (roundNum == totalRoundNum) {
            writer.close();
            roundNum = 0;
            save(robotFile);
        }
        if (epsilon > 0) {
            epsilon -= epsilonDecay;
            if (epsilon <= 0)
                epsilon = 0.0;
        }

        int[] preStateIndex = indexFor(preState);
        lookUpTable[preStateIndex[0]][preStateIndex[1]][preStateIndex[2]][preStateIndex[3]][(int)preAction] +=
                alpha * reward;
        initializePreviousIndex();
    }

    public void initializePreviousIndex() {
        Arrays.fill(preState, -0.1);
        preAction = -0.1;
    }

    public void win() {
        winsPerInterval++;
    }


    public RobocodeFileWriter getWriter() {
        return writer;
    }
}
