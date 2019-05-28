import Model.DataTuple;
import Model.NNetworkInputLoad;
import Model.TrainingData;
import Model.ValidationData;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

import java.io.*;
import java.util.zip.ZipFile;

/**
 * Loads the MNIST image data set which is neccesary to train network
 */
public class MNIST_Loader {

    public MNIST_Loader() {

    }

    /**
     * Returns a DataTuple from MNIST files such that:
     *      Let DaTaTuple be a 2-tuple (x, y) where
     *      x is an array such that each entry is an n-dimensional array (Matrix W) with 784 values.
     *          This represents the the 784 pixels in each image. Therefore, x can be abstractly represented
     *          by a 28x28 matrix W. This will become more important in the NeuralNetwork where we can manipulate
     *          this convenient representation using linear algebra in order to simplify operations.
     *
     *      y is an n-dimensional array (Vector V) where each entry is a single digit range(0-9)
     *
     */
    public DataTuple loadDataTrainingData(String dataFile, String labelFile) throws IOException {
        // read in from zip file
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFile)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        int k = 0;
        if (numberOfItems > 30000) {
            // TODO remove me after testing
            numberOfItems = 10000;
            k = 1;
        }

        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFile)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();
        if (k==1) {
            numberOfLabels = numberOfItems;
        }

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        INDArray labelVector;

        float[] labels = new float[numberOfLabels];
        INDArray[] imageMatrices = new INDArray[numberOfItems];

        assert numberOfItems == numberOfLabels;

        // get each item W and put it in tuple with its label
        // create array one row at a time
        for(int i = 0; i < numberOfItems; i++) {
            double[][] imgData = new double[nRows][nCols];
            labels[i] = labelInputStream.readUnsignedByte();

            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    imgData[r][c] = dataInputStream.readUnsignedByte();
                }
            }

            // place the image matrix W into collection
            imageMatrices[i] =  Nd4j.create(imgData);
        }


        labelVector = Nd4j.create(labels);

        dataInputStream.close();
        labelInputStream.close();
        return new DataTuple(imageMatrices, labelVector, numberOfLabels);

    }

    /**
     * Returns a tuple containing training data, validation data, and test data.
     * Alters the data to be more convenient to use for neural networks
     *
     * Training data is a list that contains 50,000 2-tuples (x,y). X is 784-dimensional numpy.ndarray
     * y is 10-dimensional numpy.ndarray
     *
     * Validation and Test data the same except 10,000
     *
     * Let TrainingData be a list containing 50,000 2-tuples X = (x,y) such that
     *      { x | [x] is 784-dimensional array corresponding to the 28x28 image Matrix }
     *      { y | [y] is 10-dimensional array corresponding to the unit vector for the correct image }
     *
     * Let ValidationData and TestData both be lists containing 10,000 2-tuples X' = (x,y') such that
     *      { y' | 0 >= y' > 10 | representing the digit value of x }
     *
     *
     *
     */
    public NNetworkInputLoad dataWrapper(String trData, String trLabel, String teData, String teLabel) throws IOException {

        // load raw data from MNIST
        DataTuple rawTrainingData = loadDataTrainingData(trData, trLabel);
        DataTuple rawTestingData = loadDataTrainingData(teData, teLabel);

        // image vector formatting
        INDArray[] trainingImageVectors = formatImageVector(rawTrainingData);
        INDArray[] testingImageVectors = formatImageVector(rawTestingData);

        // label vector formatting
        INDArray[] trainingLabelVectors = new INDArray[rawTrainingData.getLabels()];
        double[] correctLabels = new double[rawTestingData.getLabels()];

        // iterate over the y Vector in training data
        INDArray trainingLabels = rawTrainingData.getY();
        NdIndexIterator iter = new NdIndexIterator(trainingLabels.rows(), trainingLabels.columns());
        int trainingIndex = 0;
        while (iter.hasNext()) {
            int[] nextIndex = iter.next();
            double nextVal = trainingLabels.getDouble(nextIndex);

            // format y into V such that
            // V == [a_0, ..., a_9] such that a_i = { 1 | i = correct digit representing M , 0 otherwise }
            trainingLabelVectors[trainingIndex] = vectorizedResults(nextVal);
            trainingIndex++;
        }

        INDArray testingLabels = rawTestingData.getY();
        NdIndexIterator iter2 = new NdIndexIterator(testingLabels.rows(), testingLabels.columns());
        int testingIndex = 0;
        while (iter2.hasNext()) {
            int[] nextIndex = iter2.next();
            double nextVal = testingLabels.getDouble(nextIndex);

            correctLabels[testingIndex] = nextVal;
            testingIndex++;
        }


        // ensure there is an equal amount of image vectors to label vectors
        assert trainingImageVectors.length == trainingIndex;
        assert testingImageVectors.length == testingIndex;

        // encapsulate data in a package for export
        // training data
        TrainingData[] formattedTrainingData = new TrainingData[trainingIndex];
        for (int i = 0; i < trainingIndex; i++) {
            // create the list of 2-tuples
            formattedTrainingData[i] = new TrainingData(trainingImageVectors[i], trainingLabelVectors[i]);
        }

        // validation & testing data
        ValidationData[] formattedValidationData = new ValidationData[testingIndex];
        ValidationData[] formattedTestingData = new ValidationData[testingIndex];
        for (int i = 0; i < testingIndex; i++) {
            // create the list of 2-tuples
            formattedValidationData[i] = new ValidationData(testingImageVectors[i], (int)correctLabels[i]);
            formattedTestingData[i] = new ValidationData(testingImageVectors[i], (int)correctLabels[i]);
        }

        return new NNetworkInputLoad(formattedTrainingData, formattedValidationData, formattedTestingData);
    }

    // format x such into M such that
    // Let M = [a_11, ..., a_mn] such that (a_ij) elementOf( R^(m*n) )
    // abstractly, M is a matrix representing 28x28 image
    private INDArray[] formatImageVector(DataTuple rawData) {

        INDArray[] trainingImageVectors = new INDArray[rawData.getX().length];
        int i = 0;

        for (INDArray M : rawData.getX()) {
            // reshape M into (784 x 1) Vector
            trainingImageVectors[i] = M.reshape(784, 1);
            i++;
        }

        return trainingImageVectors;
    }


    /**
     * Returns a 10-dimensional unit vector with a 1.0 in the ith position and zeros elsewhere.
     * Used to convert a 0-9 digit into a corresponding desired output from the nnetwork
     */
    public INDArray vectorizedResults(double i) {
        INDArray v = Nd4j.zeros(10,1);
        v.putScalar((int)i,0,1.0);
        return v;
    }
}
