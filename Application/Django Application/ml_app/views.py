from django.shortcuts import render, redirect
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
from torchvision import models
import shutil
from PIL import Image as pImage
import time
from django.conf import settings
from .forms import VideoUploadForm


index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

class Model(nn.Module):

    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))


class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length=60,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        for i,frame in enumerate(self.frame_extract(video_path)):
            #if(i % a == first_frame):
            faces = face_recognition.face_locations(frame)
            try:
              top,right,bottom,left = faces[0]
              frame = frame[top:bottom,left:right,:]
            except:
              pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        """
        for i,frame in enumerate(self.frame_extract(video_path)):
            if(i % a == first_frame):
                frames.append(self.transform(frame))
        """        
        # if(len(frames)<self.count):
        #   for i in range(self.count-len(frames)):
        #         frames.append(self.transform(frame))
        #print("no of frames", self.count)
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    # This image is not used
    # cv2.imwrite(os.path.join(settings.PROJECT_DIR, 'uploaded_images', video_file_name+'_convert_2.png'),image*255)
    return image

def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype('uint8'))
    plt.show()


def predict(model,img,path = './', video_file_name=""):
    fmap,logits = model(img.to(device))
    img = im_convert(img[:,-1,:,:,:], video_file_name)
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _,prediction = torch.max(logits,1)
    return int(prediction.item())

def plot_heat_map(i, model, img, path = './', video_file_name=''):
  fmap,logits = model(img.to(device))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  #out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  predict = out.reshape(h,w)
  predict = predict - np.min(predict)
  predict_img = predict / np.max(predict)
  predict_img = np.uint8(255*predict_img)
  out = cv2.resize(predict_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img[:,-1,:,:,:], video_file_name)
  result = heatmap * 0.5 + img*0.8*255
  # Saving heatmap - Start
  heatmap_name = video_file_name+"_heatmap_"+str(i)+".png"
  image_name = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
  cv2.imwrite(image_name,result)
  # Saving heatmap - End
  result1 = heatmap * 0.5/255 + img*0.8
  r,g,b = cv2.split(result1)
  result1 = cv2.merge((r,g,b))
  return image_name

def load_local_model(sequence_length):
    model_path = get_accurate_model(sequence_length)
    if not model_path or not os.path.exists(model_path):
        return None, None

    model = Model(2).to(device)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    try:
        model.load_state_dict(checkpoint, strict=False)
    except RuntimeError:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, model_path

def load_local_metrics():
    metrics_candidates = [
        os.path.join(settings.PROJECT_DIR, 'models', 'metrics.json'),
        os.path.join(settings.PROJECT_DIR, 'models', 'model_metrics.json'),
        os.path.join(settings.PROJECT_DIR, 'models', 'evaluation_metrics.json'),
    ]

    for metrics_path in metrics_candidates:
        if not os.path.exists(metrics_path):
            continue
        try:
            with open(metrics_path, 'r') as fh:
                raw_metrics = json.load(fh)
            return {
                'precision': raw_metrics.get('precision'),
                'recall': raw_metrics.get('recall'),
                'confusion_matrix': raw_metrics.get('confusion_matrix'),
            }
        except Exception:
            continue

    return {
        'precision': None,
        'recall': None,
        'confusion_matrix': None,
    }

def format_metric_value(value, digits=4):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value

def normalize_metrics(raw_metrics):
    confusion_matrix = raw_metrics.get('confusion_matrix')
    if isinstance(confusion_matrix, list):
        normalized_matrix = []
        for row in confusion_matrix:
            normalized_matrix.append(row if isinstance(row, list) else [row])
        confusion_matrix = normalized_matrix

    return {
        'precision': raw_metrics.get('precision'),
        'recall': raw_metrics.get('recall'),
        'confusion_matrix': confusion_matrix,
    }

def encode_image_to_data_uri(image_array):
    success, encoded_image = cv2.imencode('.png', image_array)
    if not success:
        return None
    return f"data:image/png;base64,{base64.b64encode(encoded_image).decode('utf-8')}"

def generate_fallback_prediction(video_file_path, video_file_name_only, sequence_length):
    preview_frames = extract_preview_frames(video_file_path, frame_count=max(1, min(sequence_length, 12)))
    if not preview_frames:
        return {
            "status": "error",
            "message": "Unable to extract frames from the uploaded video.",
        }

    middle_frame_uri = preview_frames[len(preview_frames) // 2]
    raw_frame = base64.b64decode(middle_frame_uri.split(',')[-1])
    frame_array = np.frombuffer(raw_frame, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        return {
            "status": "error",
            "message": "Unable to decode the uploaded video frame.",
        }

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if face_locations:
        top, right, bottom, left = face_locations[0]
    else:
        height, width = rgb_frame.shape[:2]
        size = min(height, width) // 3
        center_y, center_x = height // 2, width // 2
        top = max(0, center_y - size)
        bottom = min(height, center_y + size)
        left = max(0, center_x - size)
        right = min(width, center_x + size)

    face_roi = rgb_frame[top:bottom, left:right]
    if face_roi.size == 0:
        face_roi = rgb_frame

    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
    edge_map = cv2.Laplacian(gray_roi, cv2.CV_32F)
    edge_map = np.abs(edge_map)
    edge_map = edge_map - np.min(edge_map)
    max_value = np.max(edge_map)
    if max_value > 0:
        edge_map = edge_map / max_value
    edge_map = np.uint8(255 * edge_map)
    edge_map = cv2.resize(edge_map, (face_roi.shape[1], face_roi.shape[0]))
    face_heatmap = cv2.applyColorMap(edge_map, cv2.COLORMAP_JET)

    overlay = rgb_frame.copy()
    overlay[top:bottom, left:right] = cv2.addWeighted(
        cv2.cvtColor(face_roi, cv2.COLOR_RGB2BGR),
        0.35,
        face_heatmap,
        0.65,
        0,
    )[:, :, ::-1]

    heatmap_image = encode_image_to_data_uri(np.uint8(cv2.resize(overlay, (640, 360))))

    artifact_score = float(np.mean(edge_map))
    output = "FAKE" if artifact_score > 18.0 else "REAL"

    return {
        "status": "success",
        "model_path": None,
        "result": output,
        "heatmap_image": heatmap_image,
        "prediction_source": "heuristic fallback",
        "artifact_score": round(artifact_score, 2),
    }

def run_local_prediction(video_file_path, sequence_length, video_file_name_only):
    model, model_path = load_local_model(sequence_length)
    if model is None:
        return generate_fallback_prediction(video_file_path, video_file_name_only, sequence_length)

    inference_dataset = validation_dataset([video_file_path], sequence_length=sequence_length, transform=train_transforms)
    inference_tensor = inference_dataset[0]

    predicted_class = predict(model, inference_tensor, video_file_name=video_file_name_only)
    heatmap_path = plot_heat_map(-1, model, inference_tensor, video_file_name=video_file_name_only)

    heatmap_image = None
    if heatmap_path and os.path.exists(heatmap_path):
        with open(heatmap_path, 'rb') as fh:
            heatmap_image = f"data:image/png;base64,{base64.b64encode(fh.read()).decode('utf-8')}"

    return {
        "status": "success",
        "model_path": model_path,
        "result": "REAL" if int(predicted_class) == 1 else "FAKE",
        "heatmap_image": heatmap_image,
        "prediction_source": "trained model",
    }

# Model Selection
def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))

    for model_path in list_models:
        model_name.append(os.path.basename(model_path))

    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except IndexError:
            pass  # Handle cases where the filename format doesn't match expected

    if len(sequence_model) > 1:
        accuracy = []
        for filename in sequence_model:
            acc = filename.split("_")[1]
            accuracy.append(acc)  # Convert accuracy to float for proper comparison
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[0])
    else:
        print("No model found for the specified sequence length.")  # Handle no models found case

    return final_model


def extract_preview_frames(video_path, frame_count=12):
    preview_frames = []
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return preview_frames

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        return preview_frames

    sample_indices = np.linspace(0, max(total_frames - 1, 0), num=min(frame_count, total_frames), dtype=int)
    for frame_index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success or frame is None:
            continue

        success, encoded_frame = cv2.imencode('.png', frame)
        if not success:
            continue

        preview_frames.append(f"data:image/png;base64,{base64.b64encode(encoded_frame).decode('utf-8')}")

    capture.release()
    return preview_frames

ALLOWED_VIDEO_EXTENSIONS = set(['mp4','gif','webm','avi','3gp','wmv','flv','mkv'])

def allowed_video_file(filename):
    #print("filename" ,filename.rsplit('.',1)[1].lower())
    if (filename.rsplit('.',1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS):
        return True
    else: 
        return False
def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})

            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if allowed_video_file(video_file.name) == False:
                video_upload_form.add_error("upload_video_file","Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            
            saved_video_file = 'uploaded_file_'+str(int(time.time()))+"."+video_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

import requests
import os
import base64
from django.shortcuts import render, redirect
from django.conf import settings

def predict_page(request):
    if request.method == "GET":
        def normalize_image_ref(item):
            """Normalize various API image formats into a browser-usable src value."""
            if not item:
                return None

            # Some backends may return objects like {"image": "..."}
            if isinstance(item, dict):
                for key in ("image", "base64", "data", "frame", "url", "path"):
                    if item.get(key):
                        item = item.get(key)
                        break

            if not isinstance(item, str):
                return None

            value = item.strip()
            if not value:
                return None

            # Already browser-ready
            if value.startswith("data:image"):
                return value
            if value.startswith("http://") or value.startswith("https://") or value.startswith("/"):
                return value

            # Assume raw base64 image content
            return f"data:image/png;base64,{value}"

        # Redirect to 'home' if 'file_name' is not in session (from your original view)
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        
        video_file_path = request.session['file_name']
        sequence_length = request.session.get('sequence_length', 60)
        video_file_name = os.path.basename(video_file_path)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        
        # Production environment path adjustments (from your original view layout)
        if not settings.DEBUG:
            production_video_name = os.path.join('/home/app/staticfiles/', video_file_name.split('/')[3])
        else:
            production_video_name = video_file_name
        
        output = "PROCESSING_ERROR"
        metrics = normalize_metrics(load_local_metrics())
        local_result = {}
        artifact_score = None
        preprocessed_images = []
        faces_cropped_images = []
        heatmap_images = []
        heatmap_image = None
        preprocessed_image_uris = []
        faces_cropped_image_uris = []

        # Ensure upload_dir exists for saved images
        upload_dir = os.path.join(settings.PROJECT_DIR, 'uploaded_images')
        os.makedirs(upload_dir, exist_ok=True)

        # Always build a local preview from the current uploaded video so the page can show
        # frames for this specific upload even if the remote backend is unavailable.
        local_preview_frames = extract_preview_frames(video_file_path, frame_count=max(1, min(sequence_length, 12)))
        if local_preview_frames:
            preprocessed_image_uris = list(local_preview_frames)

        if local_preview_frames:
            for i, image_uri in enumerate(local_preview_frames):
                image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
                image_path = os.path.join(upload_dir, image_name)
                if not os.path.exists(image_path):
                    with open(image_path, "wb") as fh:
                        fh.write(base64.b64decode(image_uri.split(",")[-1]))

        # Build cropped face thumbnails from the local preview frames so the UI shows faces
        # even if the remote backend didn't return cropped faces.
        if not faces_cropped_image_uris and local_preview_frames:
            try:
                for i, image_uri in enumerate(local_preview_frames):
                    raw = base64.b64decode(image_uri.split(",")[-1])
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    faces = face_recognition.face_locations(img)
                    if not faces:
                        # try a central crop fallback if no faces detected
                        h, w = img.shape[:2]
                        cx, cy = w // 2, h // 2
                        size = min(h, w) // 3
                        top = max(0, cy - size)
                        bottom = min(h, cy + size)
                        left = max(0, cx - size)
                        right = min(w, cx + size)
                        crop = img[top:bottom, left:right]
                        success, enc = cv2.imencode('.png', crop)
                        if success:
                            b64 = base64.b64encode(enc.tobytes()).decode('utf-8')
                            uri = f"data:image/png;base64,{b64}"
                            faces_cropped_image_uris.append(uri)
                            # save to disk
                            fname = f"{video_file_name_only}_cropped_faces_{len(faces_cropped_image_uris)}.png"
                            with open(os.path.join(upload_dir, fname), 'wb') as fh:
                                fh.write(enc.tobytes())
                    else:
                        # crop each detected face (limit to 3 per frame to avoid too many)
                        for j, (top, right, bottom, left) in enumerate(faces[:3]):
                            crop = img[top:bottom, left:right]
                            if crop.size == 0:
                                continue
                            success, enc = cv2.imencode('.png', crop)
                            if not success:
                                continue
                            b64 = base64.b64encode(enc.tobytes()).decode('utf-8')
                            uri = f"data:image/png;base64,{b64}"
                            faces_cropped_image_uris.append(uri)
                            # save to disk
                            fname = f"{video_file_name_only}_cropped_faces_{len(faces_cropped_image_uris)}.png"
                            with open(os.path.join(upload_dir, fname), 'wb') as fh:
                                fh.write(enc.tobytes())
                    # stop once we have a reasonable number of cropped faces
                    if len(faces_cropped_image_uris) >= 12:
                        break
            except Exception:
                pass

        try:
            local_result = run_local_prediction(video_file_path, sequence_length, video_file_name_only)
            output = local_result.get('result', output)
            heatmap_image = local_result.get('heatmap_image', heatmap_image)
            artifact_score = local_result.get('artifact_score')
            if local_result.get('status') != 'success':
                output = local_result.get('message', output)
        except FileNotFoundError:
            output = "The uploaded video could not be read locally."
        except Exception as exc:
            output = f"Local backend error: {exc}"

        # Pack data into your original context structure layout
        # Pass data URIs directly to the template for reliable rendering
        context = {
            'preprocessed_images': preprocessed_image_uris,
            'faces_cropped_images': faces_cropped_image_uris,
            'heatmap_image': heatmap_image,
            'heatmap_images': heatmap_images,
            'original_video': production_video_name,
            'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
            'output': output,
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'confusion_matrix': metrics.get('confusion_matrix'),
            'metrics_available': any(metrics.get(key) is not None for key in ('precision', 'recall', 'confusion_matrix')),
            'prediction_source': local_result.get('prediction_source') if isinstance(local_result, dict) else None,
            'artifact_score': format_metric_value(artifact_score, 2)
        }
        
        return render(request, predict_template_name, context)
    
    
def about(request):
    return render(request, about_template_name)

def handler404(request,exception):
    return render(request, '404.html', status=404)
def cuda_full(request):
    return render(request, 'cuda_full.html')
