import weka.classifiers.Classifier;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.*;

public class DataBlock {
    private Instances instances;
    private Collection<ClassifierWrapper> classifiers;
    private String removedAttributes;
    private int attributes;
    private double dataBlockAccuracy;

    DataBlock(Instances data, int attributes) throws Exception {
        this.attributes = attributes;
        updateRemovedAttributes(data.numAttributes());
        updateInstances(data);
    }

    public void updateInstances(Instances data) throws Exception {
        instances = removeRedundantAttributes(data);
        Classifier learner = new HoeffdingTree();
        learner.buildClassifier(instances);
        if (classifiers == null)
            classifiers = new ArrayList<>();
        classifiers.add(new ClassifierWrapper(learner));
        removeOldClassifies();
    }

    private void removeOldClassifies() {
        if (classifiers.size() > Main.Data_Block_Max_Classifier) {
            ClassifierWrapper c = classifiers.stream().filter(x -> x.getLastAccuracy() <= dataBlockAccuracy).
                    sorted(Comparator.comparingDouble(ClassifierWrapper::getLastAccuracy)).findFirst().get();
            classifiers.remove(c);
        }
    }

    public void removeWorstClassier() {
        if (classifiers.size() > Main.Data_Block_Min_Classifier) {
            ClassifierWrapper c = classifiers.stream().filter(x -> x.getLastAccuracy() <= dataBlockAccuracy).
                    sorted(Comparator.comparingDouble(ClassifierWrapper::getLastAccuracy)).findFirst().get();
            classifiers.remove(c);
        }
    }

    private void updateRemovedAttributes(int dataAttribute) {
        Collection<Integer> attributesToRemove = new ArrayList<>();

        int attrNum = 0;
        boolean forceRemoving = false;

        for (int i = 0; i < dataAttribute - 1; i++) {
            if (!forceRemoving && getChance()) {
                attrNum++;
            } else {
                attributesToRemove.add(i + 1);
            }
            if (attrNum == attributes) {
                forceRemoving = true;
            }
            if (attributesToRemove.size() + attributes + 1 == dataAttribute) {
                break;
            }
        }

        StringJoiner joiner = new StringJoiner(",");
        for (int a : attributesToRemove) {
            joiner.add("" + a);
        }
        this.removedAttributes = joiner.toString();
    }

    private Instances removeRedundantAttributes(Instances data) throws Exception {
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = removedAttributes;

        Remove remove = new Remove();
        remove.setOptions(options);
        remove.setInputFormat(data);

        Instances result = Filter.useFilter(data, remove);
        result.setClassIndex(result.numAttributes() - 1);
        return result;
    }

    double getBlockAccuracy(Instances data) throws Exception {
        data = removeRedundantAttributes(data);

        double accuracy = 0;
        for (ClassifierWrapper classifier : classifiers) {
            double tt = 0;
            for (Instance instance : data) {
                if (classifier.getClassifier().classifyInstance(instance) == instance.classValue()) {
                    tt++;
                }
            }
            tt = tt / data.numInstances();
            classifier.setLastAccuracy(tt);
            classifier.setAge(classifier.getAge() + 1);
            accuracy += tt;
        }
        dataBlockAccuracy = accuracy / classifiers.size();
        return dataBlockAccuracy;
    }

    List<List<Double>> getInctancesClass(Instances data) throws Exception {
        List<List<Double>> classes = new ArrayList<>();
        data = removeRedundantAttributes(data);
        for (Instance instance : data) {
            List<Double> c = new ArrayList<>();
            for (ClassifierWrapper classifier : classifiers) {
                c.add(classifier.getClassifier().classifyInstance(instance));
            }
            classes.add(c);
        }
        return classes;
    }

    public Instances getInstances() {
        return instances;
    }

    public Collection<ClassifierWrapper> getClassifiers() {
        return classifiers;
    }

    public void setClassifiers(Collection<ClassifierWrapper> classifiers) {
        this.classifiers = classifiers;
    }

    public String getRemovedAttributes() {
        return removedAttributes;
    }

    public void setRemovedAttributes(String removedAttributes) {
        this.removedAttributes = removedAttributes;
    }

    public int getAttributes() {
        return attributes;
    }

    public void setAttributes(int attributes) {
        this.attributes = attributes;
    }

    private boolean getChance() {
        Random rand = new Random();
        int n = rand.nextInt(50) + 1;
        return n % 2 == 0;
    }


    public double getDataBlockAccuracy() {
        return dataBlockAccuracy;
    }

    public void setDataBlockAccuracy(double dataBlockAccuracy) {
        this.dataBlockAccuracy = dataBlockAccuracy;
    }


}
