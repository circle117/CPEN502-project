package ece.cpen502;

import ece.cpen502.interfaces.NeuralNetInterface;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class NeuralNet implements NeuralNetInterface {

    private int argNumInputs;
    private int argNumHidden;                   // quality of learning
    private double argLearningRate;             // speed of learning
    private double argMomentumTerm;             // speed of learning
    private double argA;
    private double argB;
    private double[][] weightsInput;
    private double[] weightsHidden;
    private double[][] deltaWeightsInput;
    private double[] deltaWeightsHidden;
    private double[] middleRes;

    /**
     * @param argNumInputs The number of inputs in your input vector
     * @param argNumHidden The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
     * @param argLearningRate The learning rate coefficient
     * @param argMomentumTerm The momentum coefficient
     * @param argA Integer lower bound of sigmoid used by the output neuron only
     * @param argB Integer upper bound of sigmoid used by the output neuron only
     */
    public NeuralNet(int argNumInputs, int argNumHidden, double argLearningRate, double argMomentumTerm,
                     double argA, double argB) {
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.argLearningRate = argLearningRate;
        this.argMomentumTerm = argMomentumTerm;
        this.argA = argA;
        this.argB = argB;
        zeroWeights();
        initializeWeights();
    }

    @Override
    public double outputFor(double[] X) {
        double res = 0.0;

        // input layer
        middleRes = new double[argNumHidden];
        for (int i=0; i<argNumHidden; i++) {
            for (int j = 0; j < argNumInputs; j++)
                middleRes[i] += weightsInput[j][i] * X[j];
            middleRes[i] += weightsInput[argNumInputs][i]*bias;
            middleRes[i] = customSigmoid(middleRes[i]);
        }

        // hidden layer
        for (int i=0; i<argNumHidden; i++)
            res += weightsHidden[i] * middleRes[i];
        res += weightsHidden[argNumHidden]+bias;
        return customSigmoid(res);
    }

    @Override
    public double train(double[] X, double argValue) {
        double y = outputFor(X);
        double error = squareDifference(y, argValue);
        backPropagation(argValue, y, X);
        return error;
    }

    @Override
    public void save(File argFile) throws IOException {
        FileWriter writer = new FileWriter(argFile);
        // Layer Name, Index (connected by -), value
        writer.write(String.join(",", "weightsLayer", "index", "value\n"));
        for (int i=0; i<weightsInput.length; i++)
            for (int j=0; j<weightsInput[0].length; j++)
                writer.write(String.join(",", "Input",
                        i + "-" + j,
                        weightsInput[i][j]+"\n"));
        for (int i=0; i<weightsHidden.length; i++)
            writer.write(String.join(",", "Hidden",
                    String.valueOf(i), weightsHidden[i]+"\n"));
        writer.close();
    }

    @Override
    public void load(String argFileName) throws IOException {
        File file = new File(argFileName);
        Scanner scanner = new Scanner(file);
        scanner.next();
        while (scanner.hasNext()) {
            String[] str = scanner.next().split(",");
            if (str[0].equals("Input")) {
                String[] index = str[1].split("-");
                weightsInput[Integer.parseInt(index[0])][Integer.parseInt(index[1])] = Double.parseDouble(str[2]);
            } else if (str[0].equals("Hidden"))
                weightsHidden[Integer.parseInt(str[1])] = Double.parseDouble(str[2]);
        }
    }

    @Override
    public double sigmoid(double x) { return 2 / (1 + Math.exp(-x)) - 1; }

    @Override
    public double customSigmoid(double x) { return (argB-argA) / (1+Math.exp(-x)) + argA; }

    @Override
    public void zeroWeights() {
        this.weightsInput = new double[argNumInputs+1][argNumHidden];
        this.deltaWeightsInput = new double[argNumInputs+1][argNumHidden];
        this.weightsHidden = new double[argNumHidden+1];
        this.deltaWeightsHidden = new double[argNumHidden+1];
    }

    @Override
    public void initializeWeights() {
        for (int i=0; i<=argNumInputs; i++)
            for (int j=0; j<argNumHidden; j++)
                weightsInput[i][j] = Math.random() - 0.5;
        for (int i=0; i<=argNumHidden; i++)
            weightsHidden[i] = Math.random() - 0.5;
    }

    public double squareDifference(double actualValue, double predictedValue) {
        return 0.5 * Math.pow(actualValue - predictedValue, 2);
    }

    /**
     * back propagation calculation
     * @param argValue target value
     * @param y predicted value
     * @param X input matrix
     */
    @Override
    public void backPropagation(double argValue, double y, double [] X) {
        // copy the previous weights
        double[] tempHidden = new double[deltaWeightsHidden.length];
        double[][] tempInput = new double[deltaWeightsInput.length][deltaWeightsInput[0].length];
        for (int i=0; i<weightsInput.length; i++)
            tempInput[i] = Arrays.copyOf(weightsInput[i], weightsInput[i].length);

        // calculate the error signal
        double errorSignal = (argValue - y) * ((-1.0/(argB-argA)) * (y-argA) * (y-argB));

        // BP for the hidden layer
        for (int i=0; i<argNumHidden; i++) {
            tempHidden[i] = argMomentumTerm * (deltaWeightsHidden[i]) + argLearningRate * errorSignal * middleRes[i];
            weightsHidden[i] += argMomentumTerm * (deltaWeightsHidden[i])
                    + argLearningRate * errorSignal * middleRes[i];
        }
        tempHidden[argNumHidden] = argMomentumTerm * (deltaWeightsHidden[argNumHidden]) +
                argLearningRate * errorSignal * bias;
        weightsHidden[argNumHidden] += tempHidden[argNumHidden];

        // calculate the error signal
        double[] errorSignals = new double[argNumHidden];
        for (int i=0; i<argNumHidden; i++)
            errorSignals[i] = weightsHidden[i] * errorSignal * (-1.0/(argB-argA) * (middleRes[i]-argA) * (middleRes[i]-argB));

        // BP for the input layer
        for (int i=0; i<argNumHidden; i++) {
            for (int j=0; j<argNumInputs; j++) {
                tempInput[j][i] = argMomentumTerm * (deltaWeightsInput[j][i])
                        + argLearningRate * errorSignals[i] * X[j];
                weightsInput[j][i] += tempInput[j][i];
            }
            tempInput[argNumInputs][i] = argMomentumTerm * (deltaWeightsInput[argNumInputs][i])
                    + argLearningRate * errorSignals[i] * bias;
            weightsInput[argNumInputs][i] += tempInput[argNumInputs][i];
        }

        // copy the weights
        deltaWeightsHidden = Arrays.copyOf(tempHidden, tempHidden.length);
        for (int i=0; i<weightsInput.length; i++)
            deltaWeightsInput[i] = Arrays.copyOf(tempInput[i], tempInput[i].length);
    }
}
