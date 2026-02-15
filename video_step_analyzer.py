import torch
import numpy as np
import cv2
from pytorchvideo.models.hub import slowfast_r50
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import urllib.request
import os # Added for path handling

# ------------------------------------------
# 1️⃣ Device Setup
# ------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------------------------------
# 2️⃣ Load SlowFast Model
# ------------------------------------------
print("Loading SlowFast Model...")
model = slowfast_r50(pretrained=True).to(device)
model.eval()
print("SlowFast Model loaded.")

# ------------------------------------------
# 3️⃣ Load Kinetics-400 Labels
# ------------------------------------------
print("Loading Kinetics-400 Labels...")
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
labels = []
with urllib.request.urlopen(KINETICS_URL) as f:
    labels = [line.decode("utf-8").strip() for line in f.readlines()]
print("Kinetics-400 Labels loaded.")

# ------------------------------------------
# 4️⃣ Map to Custom Chef Steps
# ------------------------------------------
STEP_MAPPING = {
    "cutting vegetables": "Chop",
    "stirring": "Stir",
    "frying food": "Fry",
    "boiling water": "Boil",
    "pouring water": "Add Liquid",
    "arranging food": "Plate",
    "cooking chicken": "Fry",
    "frying vegetables": "Fry",
    "breading or breadcrumbing": "Bread",
    "scrambling eggs": "Stir"
}

# ------------------------------------------
# 5️⃣ Video Loader
# ------------------------------------------
def load_video_frames(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames found in video: " + video_path)

    frames = np.stack(frames)
    frames = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0
    return frames


# ------------------------------------------
# 6️⃣ Prepare SlowFast Input
# ------------------------------------------
def pack_pathway(frames):
    alpha = 4
    fast = frames
    slow = frames[:, ::alpha, :, :]
    return [
        slow.unsqueeze(0).to(device),
        fast.unsqueeze(0).to(device)
    ]


# ------------------------------------------
# 7️⃣ Extract Step Sequence
# ------------------------------------------
def get_step_sequence(video_path, clip_size=32):
    print(f"Processing video: {video_path}")
    frames = load_video_frames(video_path)
    T = frames.shape[1]

    steps = []

    for start in range(0, T - clip_size, clip_size):
        clip = frames[:, start:start+clip_size, :, :]
        inputs = pack_pathway(clip)

        with torch.no_grad():
            output = model(inputs)

        pred_class = torch.argmax(output, dim=1).item()
        action_label = labels[pred_class]
        print(f"  Predicted action label: {action_label}") # Added for debugging

        chef_step = STEP_MAPPING.get(action_label, "Other")
        steps.append(chef_step)

    return steps


# ------------------------------------------
# 8️⃣ Compare Step Sequences
# ------------------------------------------
def compare_step_sequences(chef_video, student_video):

    chef_steps = get_step_sequence(chef_video)
    student_steps = get_step_sequence(student_video)

    print("
Chef Steps:", chef_steps)
    print("Student Steps:", student_steps)

    # Convert string steps to numerical IDs for fastdtw
    unique_all_steps = list(set(chef_steps + student_steps))
    step_to_int = {step: i for i, step in enumerate(sorted(unique_all_steps))}

    chef_steps_encoded = [step_to_int[step] for step in chef_steps]
    student_steps_encoded = [step_to_int[step] for step in student_steps]

    # DTW alignment
    distance, path = fastdtw(
        chef_steps_encoded,
        student_steps_encoded,
        dist=lambda x, y: 0 if x == y else 1 # This lambda now works on integers
    )

    # Detect order mismatches
    mismatches = []
    prev_student_idx = -1

    for chef_idx, student_idx in path:
        if student_idx < prev_student_idx:
            mismatches.append(
                f"Order mismatch: Chef step '{chef_steps[chef_idx]}' out of order"
            )
        prev_student_idx = student_idx

    # Missing / Extra detection
    missing = list(set(chef_steps) - set(student_steps))
    extra = list(set(student_steps) - set(chef_steps))

    # Simple scoring
    score = 100
    score -= 10 * len(mismatches)
    score -= 15 * len(missing)
    score -= 10 * len(extra)

    score = max(score, 0)

    return {
        "chef_steps": chef_steps,
        "student_steps": student_steps,
        "order_mismatches": mismatches,
        "missing_steps": missing,
        "extra_steps": extra,
        "dtw_distance": distance,
        "final_score": score
    }


# ------------------------------------------
# Video Trimming Utility
# ------------------------------------------
def trim_video_opencv(input_path, output_path, duration=5):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found for trimming: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video file for trimming: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_to_write = int(fps * duration)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while frame_count < total_frames_to_write:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Video trimmed successfully: {output_path} (first {duration} seconds)")

# ------------------------------------------
# 9️⃣ Run Comparison
# ------------------------------------------
if __name__ == "__main__":
    # Define paths for video files
    # Assuming 'sample_data' directory exists in the same location as this script
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    sample_data_dir = os.path.join(script_dir, "sample_data")

    original_student_video = os.path.join(sample_data_dir, "student.mp4")
    output_chef_video = os.path.join(sample_data_dir, "chef.mp4")

    # Trim the student video to create a chef video for comparison
    # This creates 'chef.mp4' in 'sample_data' from the first 5 seconds of 'student.mp4'
    print("
--- Trimming chef video ---")
    trim_video_opencv(original_student_video, output_chef_video, duration=5)
    print("---------------------------")

    chef_video = output_chef_video
    student_video = original_student_video # Use the full student video for student's performance

    print("
--- Running Comparison ---")
    report = compare_step_sequences(chef_video, student_video)
    print("--------------------------")

    print("
========== FINAL REPORT ==========")
    for key, value in report.items():
        print(f"{key}: {value}")
