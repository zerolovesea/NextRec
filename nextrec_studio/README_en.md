# NextRec Studio

NextRec Studio is a Vue 3 based frontend project that helps users configure NextRec CLI training, feature, model, and prediction YAML files in a visual way.

## Highlights

- Separate pages for training / features / model / prediction to reduce config complexity
- Model analysis and parameter notes to help users understand what each model does

[](/assets/nextrec_studio_ui_en.png)

[](/assets/nextrec_studio_model_config_zh.png)

## How to Start

We provide three ways: script / Dockerfile / Docker Compose.

### Script Start

```bash
cd nextrec_studio
./start_nextrec_studio.sh
```

Open: `http://localhost:15173`

### Dockerfile Start

```bash
docker build -t nextrec-studio .
docker run --rm -p 15173:15173 nextrec-studio
```

Open: `http://localhost:15173`

### Docker Compose Start

```bash
docker compose up --build
```

Open: `http://localhost:15173`

## Usage

After finishing the configuration in the UI, download the YAML files and run:

`nextrec --mode train --train_config train_config.yaml`

`nextrec --mode predict --predict_config predict_config.yaml`

