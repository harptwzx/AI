import os
from gradio_client import Client

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError(" HF_TOKEN")

#  gradio_client 用 headers  token
client = Client(
    "imlixinyang/FlashWorld-Demo-Spark",
    headers={"Authorization": f"Bearer {hf_token}"}
)

result = client.predict(
    prompt="a beautiful sunset",
    api_name="/wzz"
)

print("生成", result)
