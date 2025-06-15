import gradio as gr
from huggingface_hub import hf_hub_download
import joblib
from transformers import AutoTokenizer
import numpy as np
from dotenv import load_dotenv
import onnxruntime as ort


class EmojiPrediction:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.thresholds = None
        self.repo_id = "ashish-001/tweet-emoji-predictor"
        self.load_model()
        self.load_required_data()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.repo_id)
        onnx_file = hf_hub_download(
            repo_id=self.repo_id,
            filename='model_quantized.onnx'
        )

        self.model = ort.InferenceSession(onnx_file)

    def load_required_data(self):
        filepath = hf_hub_download(
            repo_id=self.repo_id,
            filename='mlb_emoji_encoder.pkl'
        )
        with open(filepath, 'rb') as f:
            self.label_map = joblib.load(f)

        threshold_filepath = hf_hub_download(
            repo_id=self.repo_id,
            filename='thresholds.npy'
        )
        self.thresholds = np.load(threshold_filepath)

    def predict_emoji(self, text):
        if not len(text.strip()):
            return "", ""
        inputs = self.tokenizer(
            text,
            return_tensors="np"
        )

        # with torch.no_grad():
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        logits = self.model.run(None, onnx_inputs)[0]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        predicted_labels = (probs >= self.thresholds).astype(
            int).reshape(1, -1)

        emojis = "".join(self.label_map.inverse_transform(predicted_labels)[0])
        return emojis, f"{text} {emojis}"


emojiprediction = EmojiPrediction()

with gr.Blocks() as app:
    gr.Markdown("# Tweet/Text Emoji Predictor")
    with gr.Row(equal_height=True):
        textbox = gr.Textbox(lines=1, label="User Input",
                             placeholder="Start entering the text")
        textbox1 = gr.Textbox(label="Raw Emoji Output")
        textbox2 = gr.Textbox(label="Text with Emojis")
    textbox.change(
        fn=emojiprediction.predict_emoji,
        inputs=textbox,
        outputs=[textbox1, textbox2]
    )


if __name__ == "__main__":
    app.launch()
