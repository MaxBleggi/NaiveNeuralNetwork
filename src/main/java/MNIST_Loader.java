import Model.DataTuple;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

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

        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFile)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        // data rep. by n-dimensional array of 2-tuples, i.e. n rows, 2 columns
        INDArray data = Nd4j.zeros(numberOfItems, 2);

        INDArray labelVector;
        //INDArray imgMatrix = Nd4j.create(numberOfItems, nRows, nCols); // 3 dimensional matrix

        float[] labels = new float[numberOfLabels];
        INDArray[] imageMatrices = new INDArray[numberOfItems];

        assert numberOfItems == numberOfLabels;

        // get each item W and put it in tuple with its label
        // create array one row at a time
        for(int i = 0; i < numberOfItems; i++) {
            //MnistMatrix mnistMatrix = new MnistMatrix(nRows, nCols);
            INDArray row = Nd4j.zeros(1, 2);
            double[][] imgData = new double[nRows][nCols];
            //INDArray w = Nd4j.create(nRows, nCols);

            //mnistMatrix.setLabel(labelInputStream.readUnsignedByte());
            labels[i] = labelInputStream.readUnsignedByte();

           // System.out.println(labelInputStream.readInt());

            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                   // mnistMatrix.setValue(r, c, dataInputStream.readUnsignedByte());
                    imgData[r][c] = dataInputStream.readUnsignedByte();
                }
            }


            // place the image matrix W into collection
            //INDArray w = Nd4j.create(imgData);
            //imgMatrix = Nd4j.concat(0, imgMatrix, w);
            imageMatrices[i] =  Nd4j.create(imgData);
        }


        labelVector = Nd4j.create(labels);

        dataInputStream.close();
        labelInputStream.close();
        return new DataTuple(imageMatrices, labelVector);

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
     */
    public void dataWrapper() {


    }

    /**
     * Returns a 10-dimensional unit vector with a 1 in the jth position and zeros elsewhere.
     * Used to convert a 0-9 digit into a corresponding desired output from the nnetwork
     */
    public void vectorizedResults() {

    }
}
