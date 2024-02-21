import { Scalar } from "@tensorflow/tfjs-node-gpu";

import {
  DatasetConfig,
  ImageDataset,
  getCatsDatasetConfig,
  getDogsDatasetConfig,
} from "./dataset";
import { model } from "./model";

async function run(epochs: number, batchSize: number, modelSavePath: string) {
  const catsDatasetConfig: DatasetConfig = getCatsDatasetConfig();
  const dogsDatasetConfig: DatasetConfig = getDogsDatasetConfig();
  const dataset: ImageDataset = new ImageDataset([
    catsDatasetConfig,
    dogsDatasetConfig,
  ]);
  await dataset.initializeDataset();

  model.summary();

  const { images: trainImages, labels: trainLabels } = dataset.getTrainData();
  console.log("Training Images (Shape): " + trainImages.shape);
  console.log("Training Labels (Shape): " + trainLabels.shape);

  const validationSplit: number = 0.15;
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
  });

  const { images: testImages, labels: testLabels } = dataset.getTestData();
  const evalOutput: Scalar | Scalar[] = model.evaluate(testImages, testLabels);

  if (evalOutput instanceof Array) {
    console.log(
      `\nEvaluation result:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`,
    );
  } else {
    console.log(
      `\nEvaluation result:\n` +
        `  Loss = ${evalOutput.dataSync()[0].toFixed(3)}; ` +
        `Accuracy = ${evalOutput.dataSync()[0].toFixed(3)} `,
    );
  }

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

run(10, 4, "./model");
