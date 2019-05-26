package Model;

public class NNetworkInputLoad {
    private TrainingData[] trainingData;
    private ValidationData[] valData;
    private ValidationData[] testData;

    public NNetworkInputLoad(TrainingData[] trd, ValidationData[] vd, ValidationData[] ted) {
        this.trainingData = trd;
        this.valData = vd;
        this.testData = ted;
    }

    public TrainingData[] getTrainingData() {
        return trainingData;
    }

    public ValidationData[] getValData() {
        return valData;
    }

    public ValidationData[] getTestData() {
        return testData;
    }
}

