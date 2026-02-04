import { dumpYaml, parseYaml } from './yaml.js';

export function buildTrainConfig(trainState) {
  const targetList = trainState.data.target_list.filter((item) => item && item.trim().length > 0);
  const lossList = trainState.train.loss_list.filter((item) => item && item.trim().length > 0);
  const metricsList = trainState.train.metrics_list.filter((item) => item && item.trim().length > 0);

  const output = {
    session: { ...trainState.session },
    data: {
      path: trainState.data.path,
      format: trainState.data.format,
      target: targetList.length <= 1 ? (targetList[0] || '') : targetList,
      ...(trainState.data.id_column ? { id_column: trainState.data.id_column } : {}),
      valid_ratio: trainState.data.valid_ratio,
      random_state: trainState.data.random_state,
      streaming: trainState.data.streaming
    },
    feature_config: trainState.feature_config,
    model_config: trainState.model_config,
    dataloader: { ...trainState.dataloader },
    train: {
      optimizer: trainState.train.optimizer,
      optimizer_params: { ...trainState.train.optimizer_params },
      loss: lossList.length <= 1 ? (lossList[0] || '') : lossList,
      metrics: metricsList,
      epochs: trainState.train.epochs,
      batch_size: trainState.train.batch_size,
      shuffle: trainState.train.shuffle,
      log_interval: trainState.train.log_interval,
      use_wandb: trainState.train.use_wandb,
      use_swanlab: trainState.train.use_swanlab,
      device: trainState.train.device
    }
  };

  if (trainState.train.note && trainState.train.note.trim()) {
    output.train.note = trainState.train.note.trim();
  }

  if (trainState.train.use_wandb) {
    if (trainState.train.wandb_api) {
      output.train.wandb_api = trainState.train.wandb_api;
    }
    const { project, name } = trainState.train.wandb_kwargs || {};
    if (project || name) {
      output.train.wandb_kwargs = { ...(project ? { project } : {}), ...(name ? { name } : {}) };
    }
  }

  if (trainState.train.use_swanlab) {
    if (trainState.train.swanlab_api) {
      output.train.swanlab_api = trainState.train.swanlab_api;
    }
    const { project, name } = trainState.train.swanlab_kwargs || {};
    if (project || name) {
      output.train.swanlab_kwargs = { ...(project ? { project } : {}), ...(name ? { name } : {}) };
    }
  }

  if (Array.isArray(trainState.train.loss_params_list)) {
    const parsedParams = trainState.train.loss_params_list.map((item) => {
      if (item && item.trim()) {
        return parseYaml(item);
      }
      return {};
    });
    if (parsedParams.some((item) => Object.keys(item || {}).length > 0)) {
      output.train.loss_params = parsedParams;
    }
  } else if (trainState.train.loss_params_text && trainState.train.loss_params_text.trim()) {
    output.train.loss_params = parseYaml(trainState.train.loss_params_text);
  }

  if (trainState.train.loss_weights_text && trainState.train.loss_weights_text.trim()) {
    output.train.loss_weights = parseYaml(trainState.train.loss_weights_text);
  }

  if (trainState.export_onnx) {
    output.export_onnx = { ...trainState.export_onnx };
  }

  return dumpYaml(output);
}
