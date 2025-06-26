import streamlit as st
import openai
import cv2
import tempfile
from PIL import Image
from datetime import timedelta
import base64
import os
from io import BytesIO

# 🔐 Set your OpenAI API key (fixed in backend)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title("🚗 Crash Detection from Video")
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Helper to convert frame to base64
def encode_frame_to_base64(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Process uploaded video
if video_file:
    try:
        # Save video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("❌ Failed to read the video. Please upload a valid video file.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Fallback in case FPS is not detected

        frame_interval = 30  # Analyze every 30 frames (~1 sec if fps ≈ 30)
        accident_frames = []
        frame_count = 0
        stframe = st.empty()

        st.info("⏳ Analyzing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                try:
                    base64_img = encode_frame_to_base64(frame)

                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Does this image show a car accident? Just answer Yes or No."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                                ]
                            }
                        ],
                        max_tokens=10
                    )

                    reply = response.choices[0].message.content.strip().lower()
                    timestamp = str(timedelta(seconds=int(frame_count / fps)))

                    if "yes" in reply:
                        accident_frames.append((frame.copy(), timestamp))
                        cv2.putText(frame, "Accident Detected", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        stframe.image(frame, channels="BGR", caption=f"Accident at {timestamp}")

                except Exception as e:
                    st.error("⚠️ An error occurred while analyzing a frame.")
                    st.exception(e)
                    break

            frame_count += 1

        cap.release()
        st.success("✅ Video analysis complete.")

        if accident_frames:
            st.subheader("🕒 Accident Timestamps:")
            for _, ts in accident_frames:
                st.write(f"🔴 Accident at: {ts}")
        else:
            st.info("✅ No accidents were detected in the video.")

    except Exception as e:
        st.error("❌ Something went wrong while processing the video.")
        st.exception(e)
