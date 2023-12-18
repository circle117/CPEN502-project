package ece.cpen502;

import ece.cpen502.interfaces.NeuralNetInterface;
import robocode.RobocodeFileWriter;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class NeuralNet implements NeuralNetInterface {

    private final int argNumInputs;
    private final int argNumOutputs;
    private final int argNumHidden;                   // quality of learning
    private final double argLearningRate;             // speed of learning
    private final double argMomentumTerm;             // speed of learning
    private final double argA;
    private final double argB;
    private final boolean updateAll;
    private double[][] weightsInput;
    private double[][] weightsHidden;
    private double[][] deltaWeightsInput;
    private double[][] deltaWeightsHidden;
    private double[] middleRes;
    private int index;

    /**
     * @param argNumInputs The number of inputs in your input vector
     * @param argNumHidden The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
     * @param argLearningRate The learning rate coefficient
     * @param argMomentumTerm The momentum coefficient
     * @param argA Integer lower bound of sigmoid used by the output neuron only
     * @param argB Integer upper bound of sigmoid used by the output neuron only
     */
    public NeuralNet(int argNumInputs, int argNumHidden, int argNumOutputs, double argLearningRate, double argMomentumTerm,
                     double argA, double argB, boolean updateAll) {
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.argNumOutputs = argNumOutputs;
        this.argLearningRate = argLearningRate;
        this.argMomentumTerm = argMomentumTerm;
        this.argA = argA;
        this.argB = argB;
        this.updateAll = updateAll;
        zeroWeights();
        initializeWeights();
    }

    @Override
    public double[] outputFor(double[] X) {
        double[] res = new double[argNumOutputs];

        // input layer
        middleRes = new double[argNumHidden];
        for (int i=0; i<argNumHidden; i++) {
            for (int j = 0; j < argNumInputs; j++)
                middleRes[i] += weightsInput[j][i] * X[j];
            middleRes[i] += weightsInput[argNumInputs][i]*bias;
            middleRes[i] = customSigmoid(middleRes[i]);
        }

        // hidden layer
        for (int i=0; i<argNumOutputs; i++) {
            for (int j = 0; j < argNumHidden; j++)
                res[i] += weightsHidden[j][i] * middleRes[j];
            res[i] += weightsHidden[argNumHidden][i] + bias;
            res[i] = customSigmoid(res[i]);
        }
        return res;
    }

    @Override
    public double[] train(double[] X, double[] argValue) {
        double[] Y = outputFor(X);
        double[] errors = squareDifference(Y, argValue);
        backPropagation(argValue, Y, X);

        return errors;
    }

    @Override
    public void save(File argFile) throws IOException {
        RobocodeFileWriter writer = new RobocodeFileWriter(argFile);
        // Layer Name, Index (connected by -), value
        writer.write(String.join(",", "weightsLayer", "index", "value\n"));
        for (int i=0; i<weightsInput.length; i++)
            for (int j=0; j<weightsInput[0].length; j++)
                writer.write(String.join(",", "Input",
                        i + "-" + j, weightsInput[i][j]+"\n"));
        for (int i=0; i<weightsHidden.length; i++)
            for (int j=0; j<weightsHidden[0].length; j++)
                writer.write(String.join(",", "Hidden",
                        i + "-" + j, weightsHidden[i][j]+"\n"));
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
            } else if (str[0].equals("Hidden")) {
                String[] index = str[1].split("-");
                weightsHidden[Integer.parseInt(index[0])][Integer.parseInt(index[1])] = Double.parseDouble(str[2]);
            }
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
        this.weightsHidden = new double[argNumHidden+1][argNumOutputs];
        this.deltaWeightsHidden = new double[argNumHidden+1][argNumOutputs];
    }

    @Override
    public void initializeWeights() {
        for (int i=0; i<=argNumInputs; i++)
            for (int j=0; j<argNumHidden; j++)
                weightsInput[i][j] = Math.random() - 0.5;
        for (int i=0; i<=argNumHidden; i++)
            for (int j=0; j<argNumOutputs; j++)
                weightsHidden[i][j] = Math.random() - 0.5;
    }

    public double[] squareDifference(double[] actualValue, double[] predictedValue) {
        int begin = 0;
        int end = actualValue.length;
        if (!updateAll) {
            begin = index;
            end = index + 1;
        }
        double[] errors = new double[actualValue.length];
        for (int i=begin; i<end; i++)
            errors[i] = 0.5 * Math.pow(actualValue[i] - predictedValue[i], 2);
        return errors;
    }

    /**
     * back propagation calculation
     * @param argValue target value
     * @param Y predicted value
     * @param X input matrix
     */
    private void backPropagation(double[] argValue, double[] Y, double [] X) {
        // copy the previous weights
        double[][] tempHidden = new double[deltaWeightsHidden.length][deltaWeightsHidden[0].length];
        double[][] tempInput = new double[deltaWeightsInput.length][deltaWeightsInput[0].length];
        for (int i=0; i<weightsInput.length; i++)
            tempInput[i] = Arrays.copyOf(weightsInput[i], weightsInput[i].length);

        int begin = 0;
        int end = argNumOutputs;
        if (!updateAll) {
            begin = index;
            end = index + 1;
        }
        for (int k=begin; k < end; k++) {
            // calculate the error signal
            double errorSignal = (argValue[k] - Y[k]) * ((-1.0 / (argB - argA)) * (Y[k] - argA) * (Y[k] - argB));

            // BP for the hidden layer
            for (int i = 0; i < argNumHidden; i++) {
                tempHidden[i][k] = argMomentumTerm * (deltaWeightsHidden[i][k]) + argLearningRate * errorSignal * middleRes[i];
                weightsHidden[i][k] += tempHidden[i][k];
            }
            tempHidden[argNumHidden][k] = argMomentumTerm * (deltaWeightsHidden[argNumHidden][k]) +
                    argLearningRate * errorSignal * bias;
            weightsHidden[argNumHidden][k] += tempHidden[argNumHidden][k];

            // calculate the error signal
            double[] errorSignals = new double[argNumHidden];
            for (int i = 0; i < argNumHidden; i++)
                errorSignals[i] = weightsHidden[i][k] * errorSignal *
                        (-1.0 / (argB - argA) * (middleRes[i] - argA) * (middleRes[i] - argB));

            // BP for the input layer
            for (int i = 0; i < argNumHidden; i++) {
                for (int j = 0; j < argNumInputs; j++) {
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
            for (int i = 0; i < weightsInput.length; i++)
                deltaWeightsInput[i] = Arrays.copyOf(tempInput[i], tempInput[i].length);
        }
    }

    public void setIndex(int index) {
        this.index = index;
    }
}
