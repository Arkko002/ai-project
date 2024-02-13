import tf, { Tensor4D, Tensor3D } from "@tensorflow/tfjs-node-gpu";
import fs from "fs/promises";
import { PathLike, Dirent } from "fs";
import { join } from "path";

const TRAINING_SET_DIR = "../dataset/training_set/";
const TEST_SET_DIR = "../dataset/test_set/";
const VERIFICATION_SET_DIR = "../dataset/verification_set/";

export class DatasetConfig {
  constructor(
    public trainingSetPath: PathLike,
    public testSetPath: PathLike,
    public validationSetPath: PathLike,
    public labels: string[],
  ) {
    if (!labels.length) {
      throw new Error("No labels provided");
    }
  }
}

export const getDogsDatasetConfig = (): DatasetConfig => {
  return new DatasetConfig(
    join(TRAINING_SET_DIR, "dogs"),
    join(TEST_SET_DIR, "dogs"),
    join(VERIFICATION_SET_DIR, "dogs"),
    ["dog"],
  );
};

export const getCatsDatasetConfig = (): DatasetConfig => {
  return new DatasetConfig(
    join(TRAINING_SET_DIR, "cats"),
    join(TEST_SET_DIR, "cats"),
    join(VERIFICATION_SET_DIR, "cats"),
    ["cat"],
  );
};

type ImageTensorWithLabel = [ImageTensor, string];
type ImageTensor = Tensor3D | Tensor4D;

export class ImageDataset {
  constructor(
    private trainData: ImageTensorWithLabel[] = [],
    private testData: ImageTensorWithLabel[] = [],
    private verificationData: ImageTensorWithLabel[] = [],
    private datasetConfig: DatasetConfig,
  ) {
    console.log("Initalizing dataset");

    console.log("Finished dataset initalization");
  }

  public async initializeDataset() {
    this.trainData = await this.loadData(this.datasetConfig.trainingSetPath);
    this.testData = await this.loadData(this.datasetConfig.testSetPath);
    this.verificationData = await this.loadData(
      this.datasetConfig.validationSetPath,
    );
  }

  private async loadData(dataDir: PathLike): Promise<ImageTensorWithLabel[]> {
    console.log(`Loading data from ${dataDir}`);

    const files: Dirent[] = await fs.readdir(dataDir, { withFileTypes: true });
    const decodedImagesWithLabels: ImageTensorWithLabel[] = await Promise.all(
      files
        .filter(
          (file: Dirent) =>
            file.isFile() && this.isValidImageExtension(file.path),
        )
        .map(async (fileFiltered: Dirent) => {
          const filePath: string = fileFiltered.path;

          const buffer: Buffer = await fs.readFile(filePath);

          return this.decodeImageFromFile(buffer, filePath);
        }),
    );

    const labelsDecoded: string = decodedImagesWithLabels
      .map(
        (imageTensorWithLabel: ImageTensorWithLabel) => imageTensorWithLabel[1],
      )
      .reduce((acc: string, label: string) => acc.concat(`${label}, `));
    console.log(
      `Finished loading data from ${dataDir}, images count: ${decodedImagesWithLabels.length}, labels: ${labelsDecoded}`,
    );

    return decodedImagesWithLabels;
  }

  private async decodeImageFromFile(
    buffer: Buffer,
    filePath: string,
  ): Promise<[ImageTensor, string]> {
    const imageTensor: ImageTensor = tf.node
      .decodeImage(buffer)
      .resizeNearestNeighbor([96, 96]);

    // TODO: should file name be compared with labels?
    const label: string | undefined = this.datasetConfig.labels.find(
      (label: string) => filePath.toLocaleLowerCase().search(label) != -1,
    );
    if (!label) {
      throw new Error(`Could not find label for image ${filePath}`);
    }

    return [imageTensor, label];
  }

  private isValidImageExtension(path: string): boolean {
    const validExtensions = [".jpeg", ".png", ".bmp", ".gif"];

    return validExtensions.some((validExtension: string) => {
      if (path.toLocaleLowerCase().endsWith(validExtension)) {
        return true;
      }
    });
  }

  public getTrainData(): ImageTensorWithLabel[] {
    return this.trainData;
  }

  public getTestData(): ImageTensorWithLabel[] {
    return this.testData;
  }

  public getVerificationData(): ImageTensorWithLabel[] {
    return this.verificationData;
  }
}
