/**
 * Loads the MNIST image data set which is neccesary to train network
 */
public class MNIST_Loader {

    public MNIST_Loader() {

    }

    /**
     * Returns the MNIST data as a tuple containing training data, validation data, and test data
     *
     * TrainingData returned as a tuple with two entries. First contains actualy training images,
     * the second is the correct digit corresponding to that image
     *
     * Validation Data and Test Data are similar except each are smaller than Training Data
     */
    public void loadData() {

    }

    /**
     * Returns a tuple containing training data, validation data, and test data.
     * Alters the data to be more convenient to use for neural networks
     *
     * Training data is a list that contains 50,000 2-tuples (x,y). X is 784-dimensional numpy.ndarray
     * y is 10-dimensional numpy.ndarray
     *
     * Validation and Test data the same except 10,000
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
