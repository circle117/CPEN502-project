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
    private final double alpha;
    private final double gamma;
    private double epsilon;
    private double preAction = -1.0;
    private double[] preState = {-1.0, -1.0, -1.0, -1.0};

    public LookUpTable(int variableDimension1, int variableDimension2, int variableDimension3,
                       int variableDimension4, int variableDimension5,
                       double alpha, double gamma) {
        this.variableDimension1 = variableDimension1;
        this.variableDimension2 = variableDimension2;
        this.variableDimension3 = variableDimension3;
        this.variableDimension4 = variableDimension4;
        this.variableDimension5 = variableDimension5;
        this.alpha = alpha;
        this.gamma = gamma;
        initializeLUT();
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
        RobocodeFileWriter writerSave = new RobocodeFileWriter(argFile);
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

    public void update(double[] X, double reward, int newAction) {

        int[] newStateIndex = indexFor(X);
        int[] preStateIndex = indexFor(preState);
        double newQ = lookUpTable[newStateIndex[0]][newStateIndex[1]][newStateIndex[2]][newStateIndex[3]][newAction];
        double oldQ = lookUpTable[preStateIndex[0]][preStateIndex[1]][preStateIndex[2]][preStateIndex[3]][(int)preAction];
        lookUpTable[preStateIndex[0]][preStateIndex[1]][preStateIndex[2]][preStateIndex[3]][(int)preAction] +=
                alpha * (reward + gamma * newQ - oldQ);
    }


    public void updateTerminal(double reward){
        int[] preStateIndex = indexFor(preState);
        lookUpTable[preStateIndex[0]][preStateIndex[1]][preStateIndex[2]][preStateIndex[3]][(int)preAction] +=
                alpha * reward;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void initializeStateAction() {
        Arrays.fill(preState, -0.1);
        preAction = -0.1;
    }

    public double getPreAction() {
        return preAction;
    }
}
