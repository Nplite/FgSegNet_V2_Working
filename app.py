import cv2
import numpy as np
from keras.models import load_model
from scripts.instance_normalization import InstanceNormalization
from scripts.my_upsampling_2d import MyUpSampling2D
from scripts.FgSegNet_v2_module import loss, acc, loss2, acc2


def preprocess_frame(frame, target_size):
    """Resize and preprocess a video frame."""
    frame_resized = cv2.resize(frame, target_size)  # Resize to match the model input size
    frame_array = np.array(frame_resized, dtype=np.float32)  # Convert to float32
    frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension
    return frame_array

def postprocess_segmentation(segmentation, threshold=0.8):
    """Apply thresholding to the segmentation output."""
    segmentation[segmentation < threshold] = 0
    segmentation[segmentation >= threshold] = 1
    return segmentation

# Define paths
video_path = '/home/ai/Namdeo/FGBG/npl.mp4'  # Path to your input video
model_path = '/home/ai/Namdeo/FGBG/CDnetm/models25/baseline/mdl_highway.h5'

# Load the model with custom layers
model = load_model(model_path, custom_objects={
    'MyUpSampling2D': MyUpSampling2D,
    'InstanceNormalization': InstanceNormalization,
    'loss': loss,
    'acc': acc,
    'loss2': loss2,
    'acc2': acc2 })

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Define video writer for saving output
output_path = '/path/to/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (320, 240))

# Process video frame by frame
input_size = (320, 240)  # Width, Height
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame, input_size)

    # Predict segmentation mask
    probs = model.predict(preprocessed_frame, batch_size=1, verbose=0)
    segmentation_mask = probs[0].reshape(input_size[1], input_size[0])  # Reshape to (240, 320)

    # Apply thresholding to segmentation mask
    segmentation_mask = postprocess_segmentation(segmentation_mask)

    # Overlay segmentation mask on the original frame
    mask_overlay = (segmentation_mask * 255).astype(np.uint8)  # Convert to uint8 for visualization
    mask_overlay_colored = cv2.applyColorMap(mask_overlay, cv2.COLORMAP_JET)  # Apply color map
    combined_frame = cv2.addWeighted(cv2.resize(frame, input_size), 0.7, mask_overlay_colored, 0.3, 0)

    # Write the combined frame to the output video
    out.write(combined_frame)

    # Show the frame (optional)
    cv2.imshow("Segmentation", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()



# import numpy as np
# from keras.models import load_model
# from scripts.instance_normalization import InstanceNormalization
# # from scripts.my_upsampling_2d import MyUpSampling2D
# from scripts.my_upsampling_2d import MyUpSampling2D
# from scripts.FgSegNet_v2_module import loss, acc, loss2, acc2
# import imageio
# from PIL import Image

# def preprocess_frame(frame, target_size):
#     """Resize and preprocess a video frame."""
#     frame_resized = frame.resize(target_size, Image.BILINEAR)  # Resize to match the model input size
#     frame_array = np.array(frame_resized, dtype=np.float32)  # Convert to float32
#     frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension
#     return frame_array

# def postprocess_segmentation(segmentation, threshold=0.8):
#     """Apply thresholding to the segmentation output."""
#     segmentation[segmentation < threshold] = 0
#     segmentation[segmentation >= threshold] = 1
#     return segmentation

# def overlay_mask_on_frame(frame, mask):
#     """Overlay segmentation mask on the original frame."""
#     mask = (mask * 255).astype(np.uint8)  # Convert mask to uint8
#     mask = Image.fromarray(mask).convert("RGBA")  # Convert mask to RGBA
#     overlay = Image.new("RGBA", frame.size, (255, 0, 0, 100))  # Red transparent overlay
#     combined = Image.alpha_composite(frame.convert("RGBA"), overlay)
#     return combined

# # Paths
# video_path = '/path/to/video.mp4'  # Input video file
# output_path = '/path/to/output_video.mp4'  # Output video file
# model_path = '/home/ai/Namdeo/FGBG/CDnetm/models25/baseline/mdl_highway.h5'

# # Load the model
# model = load_model(model_path, custom_objects={
#     'MyUpSampling2D': MyUpSampling2D,
#     'InstanceNormalization': InstanceNormalization,
#     'loss': loss,
#     'acc': acc,
#     'loss2': loss2,
#     'acc2': acc2
# })



# # Read video using imageio
# reader = imageio.get_reader(video_path, 'ffmpeg')
# fps = reader.get_meta_data()['fps']
# writer = imageio.get_writer(output_path, fps=fps)

# # Process each frame
# input_size = (320, 240)  # Width, Height
# for frame in reader:
#     # Convert frame to PIL image
#     frame_pil = Image.fromarray(frame)

#     # Preprocess frame
#     preprocessed_frame = preprocess_frame(frame_pil, input_size)

#     # Predict segmentation mask
#     probs = model.predict(preprocessed_frame, batch_size=1, verbose=0)
#     segmentation_mask = probs[0].reshape(input_size[1], input_size[0])  # Reshape to (240, 320)

#     # Apply thresholding
#     segmentation_mask = postprocess_segmentation(segmentation_mask)

#     # Overlay mask on the frame
#     combined_frame = overlay_mask_on_frame(frame_pil, segmentation_mask)

#     # Write the frame to the output video
#     writer.append_data(np.array(combined_frame))

# # Close the writer
# reader.close()
# writer.close()
