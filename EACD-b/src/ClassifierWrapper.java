import weka.classifiers.Classifier;

public class ClassifierWrapper {
    private Classifier classifier;
    private int age;
    private double lastAccuracy;

    public ClassifierWrapper(Classifier classifier) {
        this.classifier = classifier;
    }

    public double getLastAccuracy() {
        return lastAccuracy;
    }

    public void setLastAccuracy(double lastAccuracy) {
        this.lastAccuracy = lastAccuracy;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }
}
