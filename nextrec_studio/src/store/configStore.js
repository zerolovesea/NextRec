import { reactive } from 'vue';

const uid = () => (crypto?.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2));

export const store = reactive({
  ui: {
    lang: 'zh'
  },
  train: {
    session: {
      id: 'my_experiment_session',
      artifact_root: 'nextrec_logs'
    },
    data: {
      path: '/path/to/training/data',
      format: 'parquet',
      target_list: ['label'],
      id_column: '',
      valid_ratio: 0.2,
      random_state: 2024,
      streaming: false
    },
    feature_config: 'feature_config.yaml',
    model_config: 'model_config.yaml',
    dataloader: {
      train_batch_size: 512,
      train_shuffle: true,
      valid_batch_size: 512,
      valid_shuffle: false,
      num_workers: 4,
      prefetch_factor: 2
    },
    train: {
      optimizer: 'adam',
      optimizer_params: {
        lr: 0.001,
        weight_decay: 0.0
      },
      loss_list: ['bce'],
      loss_params_list: [''],
      loss_weights_text: '',
      metrics_list: ['auc'],
      epochs: 10,
      batch_size: 512,
      shuffle: true,
      log_interval: 1,
      use_wandb: false,
      use_swanlab: false,
      note: '',
      wandb_api: '',
      wandb_kwargs: {
        project: '',
        name: ''
      },
      swanlab_api: '',
      swanlab_kwargs: {
        project: '',
        name: ''
      },
      device: 'cpu'
    },
    export_onnx: {
      enable: false,
      batch_size: 512,
      opset_version: 18
    }
  },
  feature: {
    dense: [
      {
        id: uid(),
        name: 'user_active_days_7',
        processor: {
          type: 'numeric',
          scaler: 'standard'
        },
        embedding: {
          input_dim: 1,
          proj_dim: 8,
          use_projection: false
        }
      }
    ],
    sparse: [
      {
        id: uid(),
        name: 'user_id',
        processor: {
          type: 'sparse',
          encode_method: 'hash',
          hash_size: 100000,
          min_freq: 1
        },
        embedding: {
          embedding_name: 'user_id',
          embedding_dim: 8,
          padding_idx: 0
        }
      }
    ],
    sequence: [
      {
        id: uid(),
        name: 'hist_item_seq',
        processor: {
          type: 'sequence',
          encode_method: 'hash',
          hash_size: 5000,
          min_freq: 1,
          max_len: 30,
          pad_value: 0,
          truncate: 'post',
          separator: ','
        },
        embedding: {
          embedding_name: 'hist_item_seq',
          vocab_size: 5000,
          max_len: 30,
          combiner: 'mean',
          embedding_dim: 8,
          padding_idx: 0
        }
      }
    ]
  },
  model: {
    model: 'deepfm',
    paramsText: 'mlp_params:\n  hidden_dims: [256, 128]\n  activation: relu\n  dropout: 0.2\n\nembedding_l1_reg: 1.0e-6\nembedding_l2_reg: 1.0e-5\ndense_l1_reg: 1.0e-6\ndense_l2_reg: 1.0e-4'
  },
  predict: {
    checkpoint_path: '/path/to/checkpoint',
    model_config: '',
    session: {
      id: ''
    },
    predict: {
      data_path: '/path/to/prediction/data',
      source_data_format: 'parquet',
      id_column: 'user_id',
      name: 'pred',
      save_data_format: 'csv',
      preview_rows: 5,
      batch_size: 512,
      num_workers: 4,
      num_processes: 1,
      device: 'cpu',
      use_onnx: false,
      streaming: true,
      chunk_size: 20000
    }
  }
});

export function addFeatureItem(list, type) {
  if (type === 'dense') {
    list.push({
      id: uid(),
      name: '',
      processor: { type: 'numeric', scaler: 'standard' },
      embedding: { input_dim: 1, proj_dim: 8, use_projection: false }
    });
    return;
  }
  if (type === 'sparse') {
    list.push({
      id: uid(),
      name: '',
      processor: { type: 'sparse', encode_method: 'hash', hash_size: 1000, min_freq: 1 },
      embedding: { embedding_name: '', embedding_dim: 8, padding_idx: 0 }
    });
    return;
  }
  list.push({
    id: uid(),
    name: '',
    processor: {
      type: 'sequence',
      encode_method: 'hash',
      hash_size: 1000,
      min_freq: 1,
      max_len: 30,
      pad_value: 0,
      truncate: 'post',
      separator: ','
    },
    embedding: {
      embedding_name: '',
      vocab_size: 1000,
      max_len: 30,
      combiner: 'mean',
      embedding_dim: 8,
      padding_idx: 0
    }
  });
}

export function removeFeatureItem(list, index) {
  list.splice(index, 1);
}
