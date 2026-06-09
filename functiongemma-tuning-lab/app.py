import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_TOKEN"] = ""
os.environ["GRADIO_OAUTH_DISABLE"] = "1"

import gradio.oauth as goauth
goauth.attach_oauth = lambda app: None
goauth._get_mocked_oauth_info = lambda: {"name": "local", "id": "0"}

from ui import build_interface

if __name__ == "__main__":
    demo = build_interface()
    demo.queue()
    demo.launch(server_name="127.0.0.1", share=False)
