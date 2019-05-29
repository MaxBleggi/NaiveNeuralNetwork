import Model.NablaPair;
import Model.TrainingData;
import Model.ValidationData;
import cern.colt.function.DoubleDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * Implementation of a classic Neural Network in Java.  It makes use of Stochastic Descent Learning for it's feedforward network.
 * Backpropagation is used used to determine the Gradients.
 */
public class NNetwork {

    private int[] sizes;
    private DoubleMatrix1D[] biases;
    private DoubleMatrix2D[] weights;
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

        this.biases = new DenseDoubleMatrix1D[this.layerCount - 1];
        this.weights = new DenseDoubleMatrix2D[this.layerCount - 1];

        // list of seeds for biases using normal distribution
        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            //biases[i] = Nd4j.ones(sizes[i+1],1);
           //weights[i] = Nd4j.ones(sizes[i+1],sizes[i]);

            // skew each bias and weight by a normally distributed random scalar
            biases[i] = gaussianDistribution( new DenseDoubleMatrix1D(sizes[i+1]) );
            weights[i] = gaussianDistribution( new DenseDoubleMatrix2D(sizes[i+1], sizes[i]) );
        }
    }

    private DenseDoubleMatrix1D gaussianDistribution(DenseDoubleMatrix1D V) {

        DenseDoubleMatrix1D prod = new DenseDoubleMatrix1D(V.size());

          // TODO test initializing rnd in loops for more initial variation
        for(int i = 0, r = V.size(); i < r; i++) {
                Random rnd = new Random();
             //   prod.put(i, j, M.getDouble(i, j) * rnd.nextGaussian());
                prod.set(i, rnd.nextGaussian());
        }

        return prod;
    }

    private DenseDoubleMatrix2D gaussianDistribution(DenseDoubleMatrix2D W) {

        DenseDoubleMatrix2D prod = new DenseDoubleMatrix2D(W.rows(), W.columns());

        // TODO test initializing rnd in loops for more initial variation
        for(int i = 0, r = W.rows(); i < r; i++) {
            for (int j = 0, c = W.columns(); j < c; j++) {
                Random rnd = new Random();
                //   prod.put(i, j, M.getDouble(i, j) * rnd.nextGaussian());
                prod.set(i, j, rnd.nextGaussian());
            }
        }

        return prod;
    }


    /**
     * Returns the output of the network if a is input
     */
    public DoubleMatrix1D feedForward(DoubleMatrix1D a) {
        DoubleMatrix1D b = a.copy();

        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            // feed forward algorithm
            // multiply the two matrices then add the matrix b.
            // Wrap around sigmoid to reduce range to [-1, 1]
           // b = sigmoid(weights[i].m(b).add(biases[i]));
            // b = W * a
            b = sigmoid( matrixAddition(matrixMult(weights[i], b), biases[i]) );
         //   weights[i].zMult(a, b);
        }

        return b;
    }

    private DoubleMatrix1D matrixAddition(DoubleMatrix1D a, DoubleMatrix1D b) {
        DoubleDoubleFunction plus = Double::sum;
        DoubleMatrix1D sum = a.copy();
        sum.assign(b, plus);
        return sum;
    }

    private DoubleMatrix2D matrixAddition(DoubleMatrix2D a, DoubleMatrix2D b) {
        DoubleDoubleFunction plus = Double::sum;
        DoubleMatrix2D sum = a.copy();
        sum.assign(b, plus);
        return sum;
    }

    private DoubleMatrix1D matrixSubtraction(DoubleMatrix1D a, DoubleMatrix1D b) {
        DoubleDoubleFunction sub = (v, v1) -> v - v1;
        DoubleMatrix1D sum = a.copy();
        sum.assign(b, sub);
        return sum;
    }

    private DoubleMatrix2D matrixSubtraction(DoubleMatrix2D a, DoubleMatrix2D b) {
        DoubleDoubleFunction sub = (v, v1) -> v - v1;
        DoubleMatrix2D sum = a.copy();
        sum.assign(b, sub);
        return sum;
    }

    private DoubleMatrix2D scalarMult(DoubleMatrix2D a, double scalar) {
        DoubleMatrix2D prod = a.copy();
        for (int i = 0, r = a.rows(); i < r; i++) {
            for (int j = 0, c = a.columns(); j < c; j++) {
                prod.set(i, j, a.get(i, j) * scalar);
            }
        }
        return prod;
    }

    private DoubleMatrix1D scalarMult(DoubleMatrix1D a, double scalar) {
        DoubleMatrix1D prod = a.copy();
        for (int i = 0, r = a.size(); i < r; i++) {
            prod.set(i, a.getQuick(i) * scalar);
        }
        return prod;
    }

    private DoubleMatrix1D matrixMult(DoubleMatrix2D a, DoubleMatrix1D b) {
        DoubleMatrix1D prod = new DenseDoubleMatrix1D(a.rows());
        a.zMult(b, prod);
        return prod;
    }

    private DoubleMatrix2D matrixMult(DoubleMatrix2D a, DoubleMatrix2D b) {
        DoubleMatrix2D prod = new DenseDoubleMatrix2D(a.rows(), b.columns());
        a.zMult(b, prod);
        return prod;
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

        DoubleMatrix1D[] nabla_b = new DoubleMatrix1D[this.layerCount - 1];
        DoubleMatrix2D[] nabla_w = new DoubleMatrix2D[this.layerCount - 1];

        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            nabla_b[i] = new DenseDoubleMatrix1D(biases[i].size());  // TODO nabla_w is redicuously expensive to initialize
            nabla_w[i] = new DenseDoubleMatrix2D(weights[i].rows(), weights[i].columns());
        }

        for (int i = 0; i < batchSize; i++) {
            NablaPair deltas = backPropagation(batch[i], nabla_b, nabla_w);
            DoubleMatrix1D[] delta_nabla_b = deltas.getNabla_b();
            DoubleMatrix2D[] delta_nabla_w = deltas.getNabla_w();

            for (int j = 0, len = this.layerCount - 1; j < len; j++) {
             //   nabla_b[j] = nabla_b[j].add(delta_nabla_b[j]);
                nabla_b[j] = matrixAddition(nabla_b[j], delta_nabla_b[j]);
            //    nabla_w[j] = nabla_w[j].add(delta_nabla_w[j]);
                nabla_w[j] = matrixAddition(nabla_w[j], delta_nabla_w[j]);
            }
        }

        for (int i = 0, len = this.layerCount - 1; i < len; i++) {
            // w - (eta / len(batch)) * nw
            //DoubleMatrix2D tmp = nabla_w[i].mul((eta / batchSize));
            DoubleMatrix2D tmp = scalarMult(nabla_w[i], (eta / batchSize));
            weights[i] = matrixSubtraction(weights[i], tmp);
           // weights[i] = weights[i].sub(tmp);
            //INDArray nwee = tmp.sub(weights[i]);
            DoubleMatrix1D tmp2 = scalarMult(nabla_b[i], (eta / batchSize));
         //   tmp = nabla_b[i].mul((eta / batchSize));
            biases[i] = matrixSubtraction(biases[i], tmp2);
           // biases[i] = biases[i].sub(tmp2);
        }
    }

    /**
     * Returns a tuple (nabla_b, nabla_w) representing the descent gradient for the cost function Cx
     * nabla_b & nabla_b are layer by layer lists of arrays
     */
    public NablaPair backPropagation(TrainingData trainingData, DoubleMatrix1D[] nabla_b, DoubleMatrix2D[] nabla_w) {
        // must duplicate, otherwise nabla_b and nabla_w will be alter which is undesired behavior
        DoubleMatrix1D[] nabla_b_2 = new DenseDoubleMatrix1D[this.layerCount - 1];
        DoubleMatrix2D[] nabla_w_2 = new DenseDoubleMatrix2D[this.layerCount - 1];

        // TODO possible bug, may need to recreate nabla
        System.arraycopy(nabla_b, 0, nabla_b_2, 0, nabla_b_2.length);
        System.arraycopy(nabla_w, 0, nabla_w_2, 0, nabla_w_2.length);

     //   for(int i = 0, len = this.layerCount - 1; i < len; i++){
     //       nabla_b_2[i] = Nd4j.create(biases[i].rows(), biases[i].columns());
    //        nabla_w_2[i] = Nd4j.create(weights[i].rows(), weights[i].columns());
     //   }

        DoubleMatrix1D activation = trainingData.getX();
        List<DoubleMatrix1D> activations = new ArrayList<DoubleMatrix1D>();
        activations.add(activation);
        List<DoubleMatrix1D> zVector = new ArrayList<DoubleMatrix1D>();


        for (int i = 0, len = this.layerCount - 1; i < len; i++) {

            // TODO shorten into 1 func and use elsewhere as well

            DoubleMatrix1D z = matrixAddition(matrixMult(weights[i], activation), biases[i]);
            //INDArray z = activation.mmul(weights[i]).add(biases[i]);

            zVector.add(z);

            activation = sigmoid(z);
            activations.add(activation);
        }

        DoubleMatrix1D delta = costDerivation(activations.get(activations.size() - 1), trainingData.getY());
        delta = hadamardProduct(delta , sigmoidPrime(zVector.get(zVector.size() - 1)));

        nabla_b_2[nabla_b_2.length - 1] = delta;
     //   nabla_w_2[nabla_w_2.length - 1] = delta.mmul(activations.get(activations.size() - 2).transpose());

        DoubleMatrix1D tmp = activations.get(activations.size() - 2);
        DoubleMatrix2D act = tmp.like2D(tmp.size(), 1);

        nabla_w_2[nabla_w_2.length - 1] = matrixMult(delta.like2D(delta.size(), 1), transpose(act));

        for (int i = nabla_b_2.length - 2; i >= 0; i-- ) {
            DoubleMatrix1D z = zVector.get(i);
            DoubleMatrix1D sp = sigmoidPrime(z);

            delta = hadamardProduct(matrixMult(transpose(weights[i + 1]), delta), sp);
            nabla_b_2[i] = delta;
           // nabla_w_2[i] = delta.mmul(activations.get(i).transpose());
            DoubleMatrix2D trans = transpose(activations.get(i).like2D(activations.get(i).size(), 1));
            nabla_w_2[i] = matrixMult(delta.like2D(delta.size(), 1), trans);
        }

        return new NablaPair(nabla_b_2, nabla_w_2);
    }

    private DoubleMatrix2D transpose(DoubleMatrix2D w) {
        return Algebra.DEFAULT.transpose(w);
    }

    /**
     * Returns the number of test inputs for which neural network outputs the correct result.
     */
    public int evaluate(ValidationData[] testData) {
        int numberCorrect = 0;

        for (ValidationData tuple : testData) {
            DoubleMatrix1D output = feedForward(tuple.getX());

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
            numberCorrect += maxOutputRow == tuple.getY() ? 1 : 0;
        }

        return numberCorrect;
    }

    /**
     * Return the vector of partial derivatives for the output activations
     */
    public DoubleMatrix1D costDerivation(DoubleMatrix1D outputActivations, DoubleMatrix1D y) {
        return matrixSubtraction(outputActivations, y);
    }

    /**
     * Sigmoid function
     */
    public DoubleMatrix1D sigmoid(DoubleMatrix1D z) {
        DoubleMatrix1D prod = z.copy();
        Function<Double, Double> lamda = (p -> 1 / (1 + Math.pow(Math.E, -p)));

        for(int i = 0, len = prod.size(); i < len; i++){
                double v = lamda.apply(prod.getQuick(i));
                prod.setQuick(i, v);
        }
        return prod;
    }

    /**
     * Derivative of Sigmoid function
     */
    public DoubleMatrix1D sigmoidPrime(DoubleMatrix1D z) {
        // let s = sigmoid(z) and I be the NxN identity matrix
        // returns s * (I - z)
        DoubleMatrix1D sig = sigmoid(z);

        Function<Double, Double> lamda = (p -> 1 - p);
        for(int i = 0, len = sig.size(); i < len; i++){
            double v = lamda.apply(sig.getQuick(i));
            sig.setQuick(i, v);
        }

        return hadamardProduct(sigmoid(z), sig);  // TODO may need to use z.cols or hamardProd
    }

    private int getMaxIndex(DoubleMatrix1D a) {
        double biggest = 0.0;
        int biggestIndex = 0;

        for (int i = 0, len = a.size(); i < len; i++) {
            double next = a.get(i);

            if (next > biggest) {
                biggest = next;
                biggestIndex = i;
            }
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
    DoubleMatrix1D hadamardProduct(DoubleMatrix1D M, DoubleMatrix1D W){
        DoubleMatrix1D prod = W.copy();

        for(int i = 0, r = W.size(); i < r; i++) {
            prod.set(i, M.get(i) * W.get(i));
        }

        return prod;
    }
}
