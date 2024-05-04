import os
from typing import Any, Dict, List, Literal, Optional, Union, get_args, get_origin

import llm
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference, get_model_specs

watsonx_api_key_env_var = "WATSONX_API_KEY"
watsonx_project_id_env_var = "WATSONX_PROJECT_ID"
watsonx_url_env_var = "WATSONX_URL"
default_instance_url = "https://us-south.ml.cloud.ibm.com"

watsonx_model_name_prefix = "watsonx/"


def get_env():
    api_key = os.environ.get(watsonx_api_key_env_var)
    if api_key is None:
        raise ValueError(
            f"Environment variable '{watsonx_api_key_env_var}' is not set."
        )

    project_id = os.environ.get(watsonx_project_id_env_var)
    if project_id is None:
        raise ValueError(
            f"Environment variable '{watsonx_project_id_env_var}' is not set."
        )

    return (api_key, project_id)


def add_model_name_prefix(model):
    return watsonx_model_name_prefix + model


def strip_model_name_prefix(model):
    return model.lstrip(watsonx_model_name_prefix)


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="watsonx")
    def watsonx():
        "Commands for working with IBM watsonx models"

    @watsonx.command(name="list-models")
    def list_models():
        for model_id in Watsonx.get_model_ids():
            print(model_id)

    @watsonx.command(name="list-model-options")
    def list_options():
        print(Watsonx.Options.list_string())

    @watsonx.command(name="list-embedding-models")
    def list_embedding_models():
        for model_id in WatsonxEmbedding.get_model_ids():
            print(model_id)


@llm.hookimpl
def register_models(register):
    for model_id in Watsonx.get_model_ids():
        register(Watsonx(model_id))


@llm.hookimpl
def register_embedding_models(register):
    for model_id in WatsonxEmbedding.get_model_ids():
        register(WatsonxEmbedding(model_id))


class Watsonx(llm.Model):
    model_id = "watsonx"

    can_stream = True

    class Options(llm.Options):
        decoding_method: Optional[Literal["sample", "greedy"]] = None
        length_penalty: Optional[Dict[str, Any]] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None
        random_seed: Optional[int] = None
        repetition_penalty: Optional[float] = None
        min_new_tokens: Optional[int] = None
        max_new_tokens: int = 100
        stop_sequences: Optional[List[str]] = None
        time_limit: Optional[int] = None
        truncate_input_tokens: Optional[int] = None

        def to_payload(self):
            payload = {}
            for attr, value in self.__dict__.items():
                if value is not None:
                    payload[attr] = value
            return payload

        @classmethod
        def list_string(cls):
            lines = []
            max_len = (
                max(len(attr_name) for attr_name in cls.__annotations__.keys()) + 1
            )
            for attr_name, attr_type in cls.__annotations__.items():
                origin = get_origin(attr_type)
                arg_names = []
                if origin is Union:
                    args = get_args(attr_type)
                    arg_names = [
                        str(arg).replace("typing.", "")
                        if hasattr(arg, "__args__")
                        else arg.__name__
                        for arg in args
                        if arg is not type(None)
                    ]
                elif hasattr(attr_type, "__args__"):
                    arg_names = [str(arg) for arg in attr_type.__args__]
                else:
                    arg_names = [attr_type.__name__.replace("typing.", "")]
                arg_str = ", ".join(arg_names) if len(arg_names) > 1 else arg_names[0]
                arg_str = f"{arg_str}" if hasattr(attr_type, "__args__") else arg_str
                line = f"{attr_name.ljust(max_len)}: {arg_str}"
                lines.append(line)
            return "\n".join(lines)

    def __init__(self, model_id):
        self.model_id = model_id
        self.url = os.environ.get(watsonx_url_env_var) or default_instance_url

    def __str__(self):
        return f"watsonx: {self.model}"

    @classmethod
    def get_models(cls):
        url = os.environ.get(watsonx_url_env_var) or default_instance_url
        specs = get_model_specs(url=url)
        models = specs["resources"]
        filtered_models = (
            model
            for model in models
            if any(func["id"] == "text_generation" for func in model["functions"])
        )
        for model in filtered_models:
            yield model

    @classmethod
    def get_model_ids(cls):
        return (add_model_name_prefix(model["model_id"]) for model in cls.get_models())

    def get_client(self):
        api_key, project_id = get_env()
        model_id = strip_model_name_prefix(self.model_id)
        return ModelInference(
            model_id=model_id,
            credentials={
                "apikey": api_key,
                "url": self.url,
            },
            project_id=project_id,
        )

    def build_chat_prompt(self, prompt, conversation):
        prompt_lines = []
        if conversation is not None:
            for prev_response in conversation.responses:
                prompt_lines.extend(
                    [
                        f"User: {prev_response.prompt.prompt}\n",
                        f"Assistant: {prev_response.text()}\n",
                    ]
                )

        prompt_lines.extend(
            [
                f"User: {prompt.prompt}\n",
                "Assistant:",
            ]
        )
        return "".join(prompt_lines)

    def execute(self, prompt, stream, response, conversation):
        client = self.get_client()

        if prompt.system:
            prompt.prompt = prompt.system + "\n\n" + prompt.prompt

        text = (
            prompt.prompt
            if not conversation
            else self.build_chat_prompt(prompt, conversation)
        )

        params = prompt.options.to_payload()

        if stream:
            return client.generate_text_stream(
                prompt=text,
                params=params,
            )
        else:
            return client.generate_text(
                prompt=text,
                params=params,
            )


class WatsonxEmbedding(llm.EmbeddingModel):
    model_id = "watsonx"

    key_env_var = "WATSONX_API_KEY"
    project_id_env_var = "WATSONX_PROJECT_ID"
    url_env_var = "WATSONX_URL"

    def __init__(self, model_id):
        self.model_id = model_id
        self.url = os.environ.get(watsonx_url_env_var) or default_instance_url

    def __str__(self):
        return f"watsonx embedding: {self.model}"

    @classmethod
    def get_models(cls):
        url = os.environ.get(watsonx_url_env_var) or default_instance_url
        specs = get_model_specs(url=url)
        models = specs["resources"]
        filtered_models = (
            model for model in models if "embedding model" in model["short_description"]
        )
        for model in filtered_models:
            yield model

    @classmethod
    def get_model_ids(cls):
        return (add_model_name_prefix(model["model_id"]) for model in cls.get_models())

    def get_client(self):
        api_key, project_id = get_env()
        model_id = strip_model_name_prefix(self.model_id)
        return Embeddings(
            model_id=model_id,
            credentials={
                "apikey": api_key,
                "url": self.url,
            },
            project_id=project_id,
        )

    def embed_batch(self, items):
        client = self.get_client()
        embeddings = client.embed_documents(texts=items)
        return embeddings
