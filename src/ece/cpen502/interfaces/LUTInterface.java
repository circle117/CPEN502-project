package ece.cpen502.interfaces;

public interface LUTInterface extends CommonInterface{

    /**
     * Initialize the look up table to all zeros
     */
    void initializeLUT();

    /**
     * A helper method that translates a vector being used to index the look up table
     * into an ordinal that can then used to access the associated loop up table element.
     * @param X The state action vector used to index the LUT
     * @return The index where this vector maps to
     */
    int[] indexFor(double[] X);

}
