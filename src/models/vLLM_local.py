from openai import OpenAI
from .Model import Model


class vLLM_local(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.enable_thinking = config["params"].get("enable_thinking")
        self.client = OpenAI(api_key=api_keys[api_pos], base_url="http://localhost:8000/v1")

    def query(self, msg):
        try:
            completion_kwargs = {
                "model": self.name,
                "temperature": self.temperature,
                # "max_tokens": self.max_output_tokens,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            }
            if self.enable_thinking is not None:
                completion_kwargs["extra_body"] = {
                    "enable_thinking": self.enable_thinking,
                    "chat_template_kwargs": {
                        "enable_thinking": self.enable_thinking
                    }
                }

            completion = self.client.chat.completions.create(**completion_kwargs)
            response = completion.choices[0].message.content
           
        except Exception as e:
            print(e)
            response = ""

        return response
