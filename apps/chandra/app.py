import time
import gradio as gr
from PIL import Image
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
manager = InferenceManager(method="vllm")
def ocr_image(image: Image.Image | None):
    if image is None:
        return "", "No image uploaded."
    start = time.perf_counter()
    batch = BatchInputItem(image=image.convert("RGB"), prompt_type="ocr_layout")
    result = manager.generate([batch])[0]
    elapsed = time.perf_counter() - start
    output = result.markdown if result.markdown else result.raw
    if result.error:
        output = f"Error during OCR.\n\n{output}"
    timing = f"{elapsed:.2f} seconds"
    return output, timing
demo = gr.Interface(
    fn=ocr_image,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=[
        gr.Textbox(label="OCR output (Markdown)", lines=20),
        gr.Textbox(label="Processing time"),
    ],
    title="Chandra OCR",
    description="Upload an image for GPU-accelerated OCR with layout.",
)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
