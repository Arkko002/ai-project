import tf, { AdamOptimizer, Sequential } from "@tensorflow/tfjs-node-gpu";

// TODO: make this configurable
const INPUT_SHAPE: number[] = [96, 96, 1];

const kernel_size: number[] = [3, 3];
const pool_size: [number, number] = [2, 2];
const first_filters: number = 32;
const second_filters: number = 64;
const third_filters: number = 128;
const dropout_conv: number = 0.3;
const dropout_dense: number = 0.3;

const model: Sequential = tf.sequential();
model.add(
  tf.layers.conv2d({
    inputShape: INPUT_SHAPE,
    filters: first_filters,
    kernelSize: kernel_size,
    activation: "relu",
  }),
);
model.add(
  tf.layers.conv2d({
    filters: second_filters,
    kernelSize: kernel_size,
    activation: "relu",
  }),
);
model.add(
  tf.layers.conv2d({
    filters: third_filters,
    kernelSize: kernel_size,
    activation: "relu",
  }),
);
// TODO: research
model.add(tf.layers.maxPooling2d({ poolSize: pool_size }));
// TODO: read dropout pdf
model.add(tf.layers.dropout({ rate: dropout_conv }));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({ units: 256, activation: "relu" }));
model.add(tf.layers.dropout({ rate: dropout_dense }));
model.add(tf.layers.dense({ units: 5, activation: "softmax" }));

// TODO: Read the paper
const optimizer: AdamOptimizer = tf.train.adam(0.0001);
// TODO: Read loss functions docs
model.compile({
  optimizer: optimizer,
  loss: "binaryCrossentropy",
  metrics: ["accuracy"],
});

export { model };
