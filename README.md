# llm-watsonx

[![PyPI](https://img.shields.io/pypi/v/llm-watsonx.svg)](https://pypi.org/project/llm-watsonx/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/h0rv/llm-watsonx/blob/main/LICENSE)

## Installation

Install this plugin in the same environment as LLM. From the current directory

```bash
llm install llm-watsonx
```

## Configuration

You will need to provide the following:

- API Key from IBM Cloud IAM: https://cloud.ibm.com/iam/apikeys
- Project ID (from watsonx.ai instance URL: https://dataplatform.cloud.ibm.com/projects//)

```bash
export WATSONX_API_KEY=
export WATSONX_PROJECT_ID=
```

- Optionally, if your watsonx instance is not in `us-south`:

```bash
export WATSONX_URL=
```

## Usage

Get list of commands:

```bash
llm watsonx --help
```

### Models

See all available models:

```bash
llm watsonx list-models
```

See all generation options:

```bash
llm watsonx list-model-options
```

#### Example

```bash
llm -m watsonx/meta-llama/llama-3-8b-instruct \
    -o temperature .4 \
    -o max_new_tokens 250 \
    "What is IBM watsonx?"
```

#### Chat Example

```bash
llm chat -m watsonx/meta-llama/llama-3-8b-instruct \
    -o max_new_tokens 1000 \
    -s "You are an assistant for a CLI (command line interface). Provide and help give unix commands to help users achieve their tasks."
```

### Embeddings

See all available models:

```bash
llm watsonx list-embedding-models
```

#### Example

```bash
cat README.md | llm embed -m watsonx/ibm/slate-30m-english-rtrvr
```
