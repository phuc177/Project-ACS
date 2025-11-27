package projetACS;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.util.Random;

/**
 * A reusable class for loading image datasets in DL4J.
 * - Automatically splits train/test
 * - Applies augmentation only on train set
 * - Normalizes data
 * - Returns ready-to-use iterators
 */
public class DataPipeline {

    private final int height;
    private final int width;
    private final int channels;
    private final int batchSize;
    private final long seed;
    private final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private final Random rng;
    

    public static class TrainTestData {
        public final DataSetIterator trainIter;
        public final DataSetIterator testIter;
        public final int numLabels;

        public TrainTestData(DataSetIterator trainIter, DataSetIterator testIter, int numLabels) {
            this.trainIter = trainIter;
            this.testIter = testIter;
            this.numLabels = numLabels;
        }
    }

    public DataPipeline(int height, int width, int channels, int batchSize, long seed) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.batchSize = batchSize;
        this.seed = seed;
        this.rng = new Random(seed);
    }

    /**
     * Load the dataset from a directory with subfolders as labels.
     * @param parentDir folder containing class subfolders
     * @param trainPerc percentage for train split (e.g., 80)
     */
    public TrainTestData load(File parentDir, double trainPerc) throws Exception {

        // ---------------------------------------------------------------
        // 1. Read all image paths
        // ---------------------------------------------------------------
        FileSplit fileSplit = new FileSplit(parentDir, allowedExtensions, rng);
        final ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);

        // train/test split
        InputSplit[] splits = fileSplit.sample(pathFilter, trainPerc, 100 - trainPerc);
        InputSplit trainData = splits[0];
        InputSplit testData = splits[1];

        // ---------------------------------------------------------------
        // 2. DATA AUGMENTATION (TRAIN ONLY)
        // ---------------------------------------------------------------
        ImageTransform trainTransform = new MultiImageTransform(rng,
                new FlipImageTransform(rng),    // horizontal flip
                new WarpImageTransform(5),    // random warp
                new ScaleImageTransform(rng, 0.9f, 1.1f), // scale 90-110%
                new RotateImageTransform(rng, 90.0f)
        );

        // No transform for test set
        ImageTransform noTransform = null;

        // ---------------------------------------------------------------
        // 3. Train RecordReader — WITH augmentation
        // ---------------------------------------------------------------
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        trainRR.initialize(trainData, trainTransform);

        int numLabels = trainRR.numLabels();

        DataSetIterator trainIter = new RecordReaderDataSetIterator(
                trainRR, batchSize, 1, numLabels
        );

        // ---------------------------------------------------------------
        // 4. Test RecordReader — NO augmentation
        // ---------------------------------------------------------------
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testData); // no transform

        DataSetIterator testIter = new RecordReaderDataSetIterator(
                testRR, batchSize, 1, numLabels
        );

        // ---------------------------------------------------------------
        // 5. NORMALIZATION (same scaler for train + test)
        // ---------------------------------------------------------------
//        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
//        scaler.fit(trainIter);
//        trainIter.setPreProcessor(scaler);
//        testIter.setPreProcessor(scaler);
        
        DataNormalization scaler = new VGG16ImagePreProcessor();
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        return new TrainTestData(trainIter, testIter, numLabels);
    }
}
