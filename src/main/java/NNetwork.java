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
import java.util.Arrays;
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
            //biases[i] = Nd4j.ones(sizes[i+1],1);
           //weights[i] = Nd4j.ones(sizes[i+1],sizes[i]);

            // skew each bias and weight by a normally distributed random scalar
            biases[i] = gaussianDistribution( Nd4j.ones(sizes[i+1],1) );
            weights[i] = gaussianDistribution( Nd4j.ones(sizes[i+1],sizes[i]) );
        }
    }

    private INDArray gaussianDistribution(INDArray M) {

        INDArray prod = Nd4j.create(M.rows(), M.columns());

          // TODO test initializing rnd in loops for more initial variation
        for(int i = 0, r = M.rows(); i < r; i++) {
            for (int j = 0, c = M.columns(); j < c; j++) {
                Random rnd = new Random();
                prod.put(i, j, M.getDouble(i, j) * rnd.nextGaussian());
            }
        }

        return prod;
    }


    /**
     * Returns the output of the network if a is input
     */
    public INDArray feedForward(INDArray a) {
        INDArray b = a.dup();

        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            // feed forward algorithm
            // multiply the two matrices then add the matrix b.
            // Wrap around sigmoid to reduce range to [-1, 1]
            b = sigmoid(weights[i].mmul(b).add(biases[i]));
        }

        return b;
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

            // TODO test shuffling data for better distribution
            fisherYatesShuffle(trainingData);

            for (int j = 0, diff = trainDataSize - miniBatchSize; j < diff; j += miniBatchSize) {
                // split array into two
               // TrainingData[] first = new TrainingData[miniBatchSize];
             //   System.arraycopy(trainingData, j, first, 0, j + miniBatchSize);

                TrainingData[] newBatch = Arrays.copyOfRange(trainingData, j, j + miniBatchSize);
                updateBatch(newBatch, eta);
            }

            System.out.println("#### eval time ####");
            int correct = evaluate(testData);
            System.out.println("Epoch " + (i + 1) + ": " + correct + "/" + testDataSize);
        }
    }

    private void fisherYatesShuffle(TrainingData[] arr) {
        Random rnd = new Random();
        for (int i = arr.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            TrainingData a = arr[index];
            arr[index] = arr[i];
            arr[i] = a;
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
            List<INDArray[]> deltas = backPropagation(batch[i], nabla_b, nabla_w);
            INDArray[] delta_nabla_b = deltas.get(0);
            INDArray[] delta_nabla_w = deltas.get(1);

            for (int j = 0, len = this.layerCount - 1; j < len; j++) {
                nabla_b[j] = nabla_b[j].add(delta_nabla_b[j]);
                nabla_w[j] = nabla_w[j].add(delta_nabla_w[j]);
            }
        }

        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            // w - (eta / len(batch)) * nw
            INDArray tmp = nabla_w[i].mul((eta / batchSize));

            weights[i] = weights[i].sub(tmp);
            //INDArray nwee = tmp.sub(weights[i]);

            tmp = nabla_b[i].mul((eta / batchSize));

            biases[i] = biases[i].sub(tmp);
        }
    }

    /**
     * Returns a tuple (nabla_b, nabla_w) representing the descent gradient for the cost function Cx
     * nabla_b & nabla_b are layer by layer lists of arrays
     */
    public List<INDArray[]> backPropagation(TrainingData trainingData, INDArray[] nabla_b, INDArray[] nabla_w) {
        // must duplicate, otherwise nabla_b and nabla_w will be alter which is undesired behavior
        INDArray[] nabla_b_2 = new INDArray[this.layerCount - 1];
        INDArray[] nabla_w_2 = new INDArray[this.layerCount - 1];

        // TODO possible bug, may need to recreate nabla
        //System.arraycopy(nabla_b, 0, nabla_b_2, 0, nabla_b_2.length);
        //System.arraycopy(nabla_w, 0, nabla_w_2, 0, nabla_w_2.length);

        for(int i = 0, len = this.layerCount - 1; i < len; i++){
            nabla_b_2[i] = Nd4j.create(biases[i].rows(), biases[i].columns());
            nabla_w_2[i] = Nd4j.create(weights[i].rows(), weights[i].columns());
        }

        INDArray activation = trainingData.getX();
        List<INDArray> activations = new ArrayList<INDArray>();
        activations.add(activation);
        List<INDArray> zVector = new ArrayList<INDArray>();


        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            INDArray z = weights[i].mmul(activation).add(biases[i]);
            //INDArray z = activation.mmul(weights[i]).add(biases[i]);

            zVector.add(z);

            activation = sigmoid(z);
            activations.add(activation);
        }

        INDArray delta = costDerivation(activations.get(activations.size() - 1), trainingData.getY());
        delta = hadamardProduct(delta , sigmoidPrime(zVector.get(zVector.size() - 1)));

        nabla_b_2[nabla_b_2.length - 1] = delta;
        nabla_w_2[nabla_w_2.length - 1] = delta.mmul(activations.get(activations.size() - 2).transpose());

        for (int i = nabla_b_2.length - 2; i >= 0; i-- ) {
            INDArray z = zVector.get(i);
            INDArray sp = sigmoidPrime(z);

            delta = hadamardProduct(weights[i + 1].transpose().mmul(delta), sp);
            nabla_b_2[i] = delta;
            nabla_w_2[i] = delta.mmul(activations.get(i).transpose());
        }

        List<INDArray[]> ret = new ArrayList<INDArray[]>();
        ret.add(nabla_b_2);
        ret.add(nabla_w_2);
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


           // for (int i = 0, rows = tuple.getX().rows(); i < rows; i++) {
                // get index the result and output rows with the highest value

              //  maxOutputRow = getMaxIndex(output, maxOutputRow);

               // maxResultRow = getMaxIndex(tuple.getX(), maxResultRow);

//                if (tuple.getX().getDouble(i, 0) > tuple.getX().getDouble(maxResultRow, 0))
//                    maxResultRow = i;
                //if (output.getDouble(i, 0) > output.getDouble(maxOutputRow, 0))
                 //   maxOutputRow = i;
          //  }
            maxOutputRow = getMaxIndex(output);


            // if they match, NNetowrk thought correctly!
            System.out.println(maxOutputRow == tuple.getY());
            numberCorrect += maxOutputRow == tuple.getY() ? 1 : 0;
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
        INDArray sig = sigmoid(z);
        INDArray ones = Nd4j.ones(sig.rows(), sig.columns()); // TODO test
        sig = ones.sub(sig);
        //sig = sig.sub(1);

        return hadamardProduct(sigmoid(z), sig);  // TODO may need to use z.cols or hamardProd
    }

    private int getMaxIndex(INDArray a) {
        NdIndexIterator iter2 = new NdIndexIterator(a.rows(), a.columns());

        double biggest = 0.0;
        int biggestIndex = 0;
        int i = 0;
        while (iter2.hasNext()) {
            int[] nextIndex = iter2.next();
            double nextVal = a.getDouble(nextIndex);

            if (nextVal > biggest) {
                biggest = nextVal;
                biggestIndex = i;
            }
            i++;
        }

        return biggestIndex;
    }

    /**
     * Schur product thm, Hadamard product of 2 definite matrices is also a definite matrix where
     *      Let A, B be two MxN matrices. The product of M and W is A such that C = [a_11, ..., a_mn] and { a_ij | M[a_ij] * W[a_ij] }
     * @param M an MxN Matrix
     * @param W
     * @return Hadamard product of two MxN matrices
     */
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
