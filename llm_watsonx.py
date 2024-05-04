import llm
import os

from ibm_watsonx_ai.foundation_models import get_model_specs, ModelInference, Embeddings

watsonx_url = "https://us-south.ml.cloud.ibm.com"


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="watsonx")
    def watsonx():
        "Commands for working with IBM watsonx models"

    @watsonx.command(name="list-models")
    def list_models():
        for model in Watsonx.get_models():
            print(model["model_id"])

    @watsonx.command(name="list-embedding-models")
    def list_embedding_models():
        for model in WatsonxEmbedding.get_models():
            print(model["model_id"])


@llm.hookimpl
def register_models(register):
    for model in Watsonx.get_models():
        register(Watsonx(model["model_id"]))


@llm.hookimpl
def register_embedding_models(register):
    for model in WatsonxEmbedding.get_models():
        register(WatsonxEmbedding(model["model_id"]))


class Watsonx(llm.Model):
    model_id = "watsonx"

    needs_key = "watsonx"
    key_env_var = "WATSONX_API_KEY"
    needs_project_id = "watsonx_project_id"
    project_id_env_var = "WATSONX_PROJECT_ID"

    can_stream = True

    def __init__(self, model_id, chat=False):
        self.model_id = model_id
        self.chat = chat

    @staticmethod
    def get_models():
        specs = get_model_specs(url=watsonx_url)
        models = specs["resources"]
        for model in models:
            is_text_model = any(
                [True for func in model["functions"] if func["id"] == "text_generation"]
            )
            if not is_text_model:
                continue
            yield model

    def get_client(self):
        api_key = os.environ.get(self.key_env_var)
        if api_key is None:
            raise ValueError(f"Environment variable '{self.key_env_var}' is not set.")

        project_id = os.environ.get(self.project_id_env_var)
        if project_id is None:
            raise ValueError(
                f"Environment variable '{self.project_id_env_var}' is not set."
            )

        return ModelInference(
            model_id=self.model_id,
            credentials={
                "apikey": api_key,
                "url": watsonx_url,
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
        return prompt_lines

    def execute(self, prompt, stream, response, conversation):
        client = self.get_client()

        lines = [prompt.prompt]
        if self.chat:
            lines = self.build_chat_prompt(prompt, conversation)

        p = "".join(lines)

        if stream:
            return client.generate_text_stream(
                prompt=p,
                params=None,
            )
        else:
            return client.generate_text(
                prompt=p,
                params=None,
            )


class WatsonxEmbedding(llm.EmbeddingModel):
    model_id = "watsonx"

    needs_key = "watsonx"
    key_env_var = "WATSONX_API_KEY"
    needs_project_id = "watsonx_project_id"
    project_id_env_var = "WATSONX_PROJECT_ID"

    can_stream = True

    def __init__(self, model_id):
        self.model_id = model_id

    @staticmethod
    def get_models():
        specs = get_model_specs(url=watsonx_url)
        models = specs["resources"]
        for model in models:
            if "embedding model" not in model["short_description"]:
                continue
            yield model

    def get_client(self):
        api_key = os.environ.get(self.key_env_var)
        if api_key is None:
            raise ValueError(f"Environment variable '{self.key_env_var}' is not set.")

        project_id = os.environ.get(self.project_id_env_var)
        if project_id is None:
            raise ValueError(
                f"Environment variable '{self.project_id_env_var}' is not set."
            )

        return Embeddings(
            model_id=self.model_id,
            credentials={
                "apikey": api_key,
                "url": watsonx_url,
            },
            project_id=project_id,
        )

    def embed_batch(self, items):
        client = self.get_client()

        embeddings = client.embed_documents(texts=items)

        return embeddings
