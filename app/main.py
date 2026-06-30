import os

import gradio as gr
import uvicorn

from app.api.routes import create_api_app
from app.ui import _auth, build_ui

app = create_api_app()
demo = build_ui()
app = gr.mount_gradio_app(app, demo, path="/", auth=_auth())

if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", os.getenv("SURYA_UI_PORT", "7860")))
    uvicorn.run(app, host=host, port=port)
