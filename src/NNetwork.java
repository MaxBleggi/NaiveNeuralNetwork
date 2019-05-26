import java.util.List;

public class NNetwork {

    private List sizes;
    private List biases;
    private List weights;
    private int layerCount;

    /**
     * The list 'sizes' corresponds to the neuron count in the nnetwork's layers.
     * In more concise terms
     *      Let e be in l where l = List()
     *      indexOf(e) = corresponding layer in nnetwork
     *      e = number of neurons in each layer of the graph
     *
     * Weights of edges are initialized using the Gaussian distribution as a bias with a mean of 0 and variance of 1.
     *
     * First layer is input, no biases are needed for these neurons.
     *
     */
    public NNetwork(List sizes) {

    }

    /**
     * Returns the output of the network if a is input
     */
    public void feedForward() {

    }

    /**
     * This implementation utilizes the Stochastic variant of Gradient Descent. Which is meant to
     * parse the entire set of data into smaller batches
     *
     * trainingData is a list of tuples (x,y) s.t.
     *      x: input
     *      y: desired output
     *
     * Network will be evaluated against each batch of test data after each epoch. Making progress tracking more simple,
     * at the sarcice of a non-trivial performance hit
     */
    public void stochGradientDescent() {

    }

    /**
     * Updates the network's weights and biases by apply Gradient Descent via backpropagation to one batch at a time
     * Each batch is a list of tuples (x,y) and eta which represents the learning rate
     */
    public void updateBatch() {

    }

    /**
     * Returns a tuple (nabla_b, nabla_w) representing the descent gradient for the cost function Cx
     * nabla_b & nabla_b are layer by layer lists of arrays
     */
        public void backPropagation() {

    }

    /**
     * Returns the number of test inputs for which neural network outputs the correct result.
     */
    public void evaluate() {

    }

    /**
     * Return the vector of partial derivatives for the output activations
     */
    public void costDerivation() {

    }

    /**
     * Sigmoid function
     */
    public void sigmoid() {

    }

    /**
     * Derivative of Sigmoid function
     */
    public void sigmoidPrime() {

    }
}
