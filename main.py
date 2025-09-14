import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import soundfile as sf
from torchaudio import transforms

class CheckAudio(nn.Module):
    def __init__(self, num_classes=8):
        super(CheckAudio, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8,8))
        )
        self.flatten = nn.Flatten()
        self.second = nn.Sequential(
            nn.Linear(64*8*8,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,num_classes)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.flatten(x)
        x = self.second(x)
        return x


sample_rate = 22050
max_len = 500
transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CheckAudio(num_classes=8)
model.load_state_dict(torch.load("model_emotion.pth", map_location=device))
model.to(device)
model.eval()

labels_list = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
index_to_labels = {i: label for i, label in enumerate(labels_list)}


def change_audio(waveform, sr):
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(waveform)

    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if sr != sample_rate:
        resample = transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resample(waveform)

    spec = transform(waveform)

    if spec.shape[-1] > max_len:
        spec = spec[:, :, :max_len]
    elif spec.shape[-1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[-1]))

    return spec

app = FastAPI()

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Пустой файл")

        wf, sr = sf.read(io.BytesIO(data), dtype="float32")
        wf = torch.from_numpy(wf).T if not isinstance(wf, torch.Tensor) else wf

        spec = change_audio(wf, sr)
        spec = spec.unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = torch.argmax(y_pred, dim=1).item()
            pred_class = index_to_labels.get(pred_ind, "Unknown")

        return {"index": pred_ind, "emotion": pred_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9001)
