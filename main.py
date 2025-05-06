from flask import Flask, request, jsonify
from flask_cors import CORS
from roboflow import Roboflow
import supervision as sv
import cv2
import io
import logging
import cloudinary
import cloudinary.uploader
import tempfile
import os
import gc
import concurrent.futures

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cloudinary configuration
cloudinary.config(
    cloud_name="dximtsuzo",
    api_key="556435271379951",
    api_secret="WL4A9wfF9pHNL2ItDjHcrWd6Mn0",
    secure=True
)

# Initialize Roboflow
try:
    rf = Roboflow(api_key="PPuv9P2fZaxGsxDvETOM")
    project = rf.workspace().project("poultarypro")
    model = project.version(4).model
    logger.info("Roboflow model loaded successfully")
except Exception as e:
    logger.error(f"Failed to initialize Roboflow: {str(e)}")
    raise

# Helper function to process a single frame
def process_frame(frame, frame_count, model, label_annotator, mask_annotator):
    # Save frame temporarily for Roboflow inference
    temp_frame_fd, temp_frame_path = tempfile.mkstemp(suffix='.jpg')
    try:
        cv2.imwrite(temp_frame_path, frame)
        # Perform inference
        result = model.predict(temp_frame_path, confidence=40).json()
        labels = [item["class"] for item in result["predictions"]]
        detections = sv.Detections.from_inference(result)

        # Annotate frame
        annotated_frame = mask_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        return frame_count, annotated_frame, result['predictions']  # Return results
    except Exception as e:
        logger.error(f"Error processing frame {frame_count}: {str(e)}")
        return frame_count, None, None
    finally:
        os.close(temp_frame_fd)
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    logger.info("Health check requested")
    return jsonify({'status': 'Server is running'}), 200

@app.route('/predict_video', methods=['POST'])
def predict_video():
    logger.info("Predict video endpoint called")
    if 'video' not in request.files:
        logger.error("No video provided")
        return jsonify({'error': 'No video provided'}), 400

    file = request.files['video']
    if not file.filename.endswith(('.mp4', '.avi')):
        logger.error("Unsupported video format")
        return jsonify({'error': 'Unsupported video format. Use .mp4 or .avi'}), 400

    # Save video to temporary file (required for reliable VideoCapture)
    temp_video_fd, temp_video_path = tempfile.mkstemp(suffix='.mp4')
    try:
        file.save(temp_video_path)
        logger.info("Video saved to temporary file")
        
        # Open video with VideoCapture
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error("Could not open video file")
            raise Exception("Could not open video file")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"Video properties: {frame_width}x{frame_height}, {fps} FPS")

        # Initialize annotators
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        # Create temporary file for output video
        temp_output_fd, temp_output_path = tempfile.mkstemp(suffix='.mp4')
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                raise Exception("Could not initialize video writer")

            frame_count = 0
            predictions = []  # List to store predictions for all frames

            # Executor to process frames concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Submit task for each frame
                    futures.append(executor.submit(process_frame, frame, frame_count, model, label_annotator, mask_annotator))
                    frame_count += 1

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    count, annotated_frame, result = future.result()
                    if annotated_frame is not None:
                        out.write(annotated_frame)
                    if result is not None:
                        predictions.append({
                            'frame': count,
                            'predictions': result,
                            'labels': [item["class"] for item in result]
                        })

            # Release resources
            cap.release()
            out.release()
            del out
            logger.info(f"Processed {frame_count} frames")

            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(temp_output_path, resource_type="video")
            public_id = upload_result.get("public_id")
            logger.info(f"Video uploaded to Cloudinary with public_id: {public_id}")

            if not public_id:
                raise Exception("Cloudinary upload failed")

            return jsonify({
                'frame_count': frame_count,
                'predictions': predictions,
                'public_id': public_id
            })

        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}")
            raise
        finally:
            os.close(temp_output_fd)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

    except Exception as e:
        logger.error(f"Error reading video: {str(e)}")
        return jsonify({'error': str(e)}), 400
    finally:
        os.close(temp_video_fd)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        gc.collect()

if __name__ == '__main__':
    logger.info("Starting Flask server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)
