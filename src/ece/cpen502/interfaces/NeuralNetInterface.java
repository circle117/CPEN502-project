package ece.cpen502.interfaces;

public interface NeuralNetInterface extends CommonInterface{

    double bias = 1.0;

    /**
     * Return a bipolar sigmoid of the input x
     * @param x The input
     * @return f(x) = 2/(1+e^(-x)) - 1
     */
    double sigmoid(double x);

    /**
     * This method implements a general sigmoid with asymptotes bounded by (a, b)
     * @param x The input
     * @return f(x) = (b-a)/(1+e^(-x)) - (-a)
     */
    double customSigmoid(double x);

    /**
     * Initialize the weights to random values
     * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
     * Like wise for hidden units. For say 2 hidden units which are stored in an array.
     * [0] & [1] are the hidden & [2] the bias.
     * We also initialize the last weight change arrays. This is to implement the alpha term.
     */
    void initializeWeights();

    /**
     * Initialize the weights to 0.
     */
    void zeroWeights();

    /**
     * calculate the square difference error
     * @param actualValue
     * @param calculatedValue
     * @return 0.5* (y - c)^2
     */
    double[] squareDifference(double[] actualValue, double[] calculatedValue);

}
