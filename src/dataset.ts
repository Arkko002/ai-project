import tf, {
  Tensor4D,
  Tensor3D,
  Rank,
  Tensor,
} from "@tensorflow/tfjs-node-gpu";
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

export interface ImageDatasetWithLabels {
  images: ImageTensor;
  labels: Tensor<Rank>;
}

type ImageTensorWithLabel = [ImageTensor, string];
type ImageTensor = Tensor3D | Tensor4D;

export class ImageDataset {
  constructor(
    private datasetConfig: DatasetConfig[],
    private trainData: ImageTensorWithLabel[] = [],
    private testData: ImageTensorWithLabel[] = [],
    private verificationData: ImageTensorWithLabel[] = [],
  ) {}

  public async initializeDataset() {
    console.log("Initalizing dataset");

    for (const datasetConfig of this.datasetConfig) {
      this.trainData.concat(
        await this.loadData(
          datasetConfig.trainingSetPath,
          datasetConfig.labels,
        ),
      );

      this.testData.concat(
        await this.loadData(datasetConfig.testSetPath, datasetConfig.labels),
      );

      this.verificationData.concat(
        await this.loadData(
          datasetConfig.validationSetPath,
          datasetConfig.labels,
        ),
      );
    }

    console.log("Finished dataset initalization");
  }

  private async loadData(
    dataDir: PathLike,
    labels: string[],
  ): Promise<ImageTensorWithLabel[]> {
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

          return this.decodeImageFromFile(buffer, filePath, labels);
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
    labels: string[],
  ): Promise<[ImageTensor, string]> {
    const imageTensor: ImageTensor = tf.node
      .decodeImage(buffer)
      .resizeNearestNeighbor([96, 96]);

    // TODO: should file name be compared with labels?
    const label: string | undefined = labels.find(
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

  public getTrainData(): ImageDatasetWithLabels {
    const images = tf.concat(
      this.trainData.map((imageTensorWithLabel) => imageTensorWithLabel[0]),
    );

    const labels = tf.oneHot(
      tf.tensor1d(
        this.trainData.map((imageTensorWithLabel) => imageTensorWithLabel[1]),
        "string",
      ),
      3,
    );

    return {
      images,
      labels,
    };
  }

  public getTestData(): ImageDatasetWithLabels {
    const images = tf.concat(
      this.testData.map((imageTensorWithLabel) => imageTensorWithLabel[0]),
    );

    const labels = tf.oneHot(
      tf.tensor1d(
        this.testData.map((imageTensorWithLabel) => imageTensorWithLabel[1]),
        "string",
      ),
      3,
    );

    return {
      images,
      labels,
    };
  }

  public getVerificationData(): ImageDatasetWithLabels {
    const images = tf.concat(
      this.verificationData.map(
        (imageTensorWithLabel) => imageTensorWithLabel[0],
      ),
    );

    const labels = tf.oneHot(
      tf.tensor1d(
        this.verificationData.map(
          (imageTensorWithLabel) => imageTensorWithLabel[1],
        ),
        "string",
      ),
      3,
    );

    return {
      images,
      labels,
    };
  }
}
