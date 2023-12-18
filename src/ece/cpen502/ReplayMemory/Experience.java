package ece.cpen502.ReplayMemory;

public class Experience {

    public double[] state = new double[4];
    public double action;
    public double reward;
    public double[] nextState = new double[4];
}
