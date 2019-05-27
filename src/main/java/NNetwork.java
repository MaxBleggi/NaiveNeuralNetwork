import Model.TrainingData;
import Model.ValidationData;
import org.nd4j.context.Nd4jContext;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.Nd4jBlas;
import org.nd4j.*;
import org.nd4j.nativeblas.Nd4jCpu;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Implementation of a classic Neural Network in Java.  It makes use of Stochastic Descent Learning for it's feedforward network.
 * Backpropagation is used used to determine the Gradients.
 */
public class NNetwork {

    private int[] sizes;
    private INDArray[] biases;
    private INDArray[] weights;
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
    public NNetwork(int[] sizes) {
        this.layerCount = sizes.length;
        this.sizes = sizes;

        this.biases = new INDArray[this.layerCount - 1];
        this.weights = new INDArray[this.layerCount - 1];

        // list of seeds for biases using normal distribution
        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            Random rnd = new Random();
            biases[i] = Nd4j.ones(sizes[i+1],1);
            weights[i] = Nd4j.ones(sizes[i+1],sizes[1]);

            // skew each bias and weight by a normally distributed random scalar
            biases[i] = biases[i].mul(rnd.nextGaussian());
            weights[i] = weights[i].mul(rnd.nextGaussian());
        }
    }

    /**
     * Returns the output of the network if a is input
     */
    public INDArray feedForward(INDArray a) {
        INDArray b = a.dup();
        INDArray c = b.dup();

        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            // feed forward algorithm
            // multiply the two matrices then add the matrix b.
            // Wrap around sigmoid to reduce range to [-1, 1]
            // product = weights[i].mul(b).add(biases[i]);
            //b = sigmoid(product);
            //Nd4j.getEx
            c = sigmoid(weights[i].mmul(b).add(biases[i]));
        }

        return c;
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
    public void stochGradientDescent(TrainingData[] trainingData, ValidationData[] testData, int epochs, int miniBatchSize, double eta) {
        int trainDataSize = trainingData.length;
        int testDataSize = testData.length;

        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < trainDataSize - miniBatchSize; j += miniBatchSize) {
                // split array into two
                TrainingData[] first = new TrainingData[miniBatchSize];
                System.arraycopy(trainingData, j, first, 0, j + miniBatchSize);

                updateBatch(first, eta);
            }

            int correct = evaluate(testData);
            System.out.println("Epoch " + (i + 1) + ": " + correct + "/" + testDataSize);
        }
    }

    /**
     * Updates the network's weights and biases by apply Gradient Descent via backpropagation to one batch at a time
     * Each batch is a list of tuples (x,y) and eta which represents the learning rate
     */
    public void updateBatch(TrainingData[] batch, double eta) {
        int batchSize = batch.length;

        INDArray[] nabla_b = new INDArray[this.layerCount - 1];
        INDArray[] nabla_w = new INDArray[this.layerCount - 1];

        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            nabla_b[i] = Nd4j.create(biases[i].rows(), biases[i].columns());
            nabla_w[i] = Nd4j.create(weights[i].rows(), weights[i].columns());
        }

        for (int i = 0; i < batchSize; i++) {

        }
    }

    /**
     * Returns a tuple (nabla_b, nabla_w) representing the descent gradient for the cost function Cx
     * nabla_b & nabla_b are layer by layer lists of arrays
     */
    public List<INDArray[]> backPropagation(TrainingData trainingData, INDArray[] nabla_b, INDArray[] nabla_w) {
        List<INDArray> activations = new ArrayList<INDArray>();
        INDArray activation = trainingData.getX();
        List<INDArray> zVector = new ArrayList<INDArray>();

        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            INDArray z = weights[i].mmul(activation).add(biases[i]);
            zVector.add(z);

            activation = sigmoid(z);
            activations.add(activation);
        }

        INDArray delta = costDerivation(activations.get(activations.size() - 1), trainingData.getY());
        delta = hadamardProduct(delta , sigmoidPrime(zVector.get(zVector.size() - 1))); // TODO schur prod

        nabla_b[nabla_b.length - 1] = delta;
        nabla_w[nabla_w.length - 1] = delta.mmul(activations.get(activations.size() - 2).transpose());

        for (int i = nabla_b.length - 2; i >= 0; i-- ) {
            INDArray z = zVector.get(i);
            INDArray sp = sigmoidPrime(z);

            delta = hadamardProduct(weights[i + 1].transpose().mmul(delta), sp);
            nabla_b[i] = delta;
            nabla_w[i] = delta.mmul(activations.get(i).transpose());
        }

        List<INDArray[]> ret = new ArrayList<INDArray[]>();
        ret.add(nabla_b);
        ret.add(nabla_w);

        return ret;
    }

    /**
     * Returns the number of test inputs for which neural network outputs the correct result.
     */
    public int evaluate(ValidationData[] testData) {
        int numberCorrect = 0;

        for (ValidationData tuple : testData) {
            INDArray output = feedForward(tuple.getX());

            int maxResultRow = 0;
            int maxOutputRow = 0;

            for (int i = 0, rows = tuple.getX().rows(); i < rows; i++) {
                // get index the result and output rows with the highest value
                maxOutputRow = getMaxIndex(tuple.getX(), maxOutputRow);
                maxResultRow = getMaxIndex(tuple.getX(), maxResultRow);
            }

            // if they match, NNetowrk thought correctly!
            numberCorrect += maxOutputRow == maxResultRow ? 1 : 0;
        }

        return numberCorrect;
    }

    /**
     * Return the vector of partial derivatives for the output activations
     */
    public INDArray costDerivation(INDArray outputActivations, INDArray y) {
        return outputActivations.sub(y);
    }

    /**
     * Sigmoid function
     */
    public INDArray sigmoid(INDArray z) {
        return Nd4j.getExecutioner().execAndReturn(new Sigmoid(z));
    }

    /**
     * Derivative of Sigmoid function
     */
    public INDArray sigmoidPrime(INDArray z) {
        // let s = sigmoid(z) and I be the NxN identity matrix
        // returns s * (I - z)
        return sigmoid(z).mmul((Nd4j.eye(z.rows()).sub(sigmoid(z))));  // TODO may need to use z.cols or hamardProd
    }

    private int getMaxIndex(INDArray a, int comp) {
        NdIndexIterator iter2 = new NdIndexIterator(a.rows(), a.columns());
        int i = 0;
        while (iter2.hasNext()) {
            int[] nextIndex = iter2.next();
            double nextVal = a.getDouble(nextIndex);

            if (nextVal > comp) {
                return i;
            }
            i++;
        }

        return comp;
    }


    INDArray hadamardProduct(INDArray M, INDArray W){
        INDArray prod = Nd4j.create(W.rows(), W.columns());

        for(int i = 0, r = W.rows(); i < r; i++) {
            for (int j = 0, c = W.columns(); j < c; j++) {
                prod.put(i, j, M.getDouble(i, j) * W.getDouble(i, j));
            }
        }

        return prod;
    }
}
