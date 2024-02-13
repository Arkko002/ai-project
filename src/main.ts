import { layers, Sequential, gather_util } from "@tensorflow/tfjs-node-gpu";

async function run_cpu_benchmark() {
  // Create a simple model.
  const model = tf_C_CPU.sequential();
  model.add(tf_C_CPU.layers.dense({ units: 1, inputShape: [1] }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  // Generate some synthetic data for training. (y = 2x - 1)
  const xs = tf_C_CPU.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tf_C_CPU.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  // Train the model using the data.
  await model.fit(xs, ys, { epochs: 250 });

  console.log(tf_C_CPU.tensor2d([20], [1, 1]).dataSync());
}

async function run_gpu_benchmark() {
  // Create a simple model.
  const model = tf_C_GPU.sequential();
  model.add(tf_C_GPU.layers.dense({ units: 1, inputShape: [1] }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  // Generate some synthetic data for training. (y = 2x - 1)
  const xs = tf_C_GPU.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tf_C_GPU.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  // Train the model using the data.
  await model.fit(xs, ys, { epochs: 250 });

  console.log(tf_C_GPU.tensor2d([20], [1, 1]).dataSync());
}
