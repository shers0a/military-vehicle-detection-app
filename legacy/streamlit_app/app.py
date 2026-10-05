import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
import os

st.set_page_config(page_title="Military Detection App", layout="centered")
MODEL_PATH = 'best.pt'


@st.cache_resource
def load_sahi_model(path):
    if not os.path.exists(path):
        alt_path = './runs/detect/rezultat_militar/weights/best.pt'
        if os.path.exists(alt_path):
            path = alt_path
        else:
            st.error(
                f"CRITICAL ERROR: Model file '{path}' not found! Please place 'best.pt' in the same folder as app.py")
            return None

    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=path,
            confidence_threshold=0.35,
            device=0,
        )
        return detection_model
    except Exception as e:
        st.error(f"Error loading SAHI model: {e}")
        return None


def coords(image_np, model):
    result = get_sliced_prediction(
        image_np,
        model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    coord = []
    for obj in result.object_prediction_list:
        if obj.category.name != "Civilian_Vehicle":
            bbox = obj.bbox.to_xyxy()
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            coord.append([center_x, center_y])

    if len(coord) == 0:
        return np.empty((0, 2))

    return np.array(coord)


def calculate_tactical_heatmap(points_t0, points_t1, img_w, img_h):
    GSD = 0.3
    grid_size = 50
    pixelgrid = grid_size / GSD

    bins_x = int(img_w / pixelgrid)
    bins_y = int(img_h / pixelgrid)

    p0_x = points_t0[:, 0] if points_t0.size > 0 else []
    p0_y = points_t0[:, 1] if points_t0.size > 0 else []

    p1_x = points_t1[:, 0] if points_t1.size > 0 else []
    p1_y = points_t1[:, 1] if points_t1.size > 0 else []

    heatmap_t0, _, _ = np.histogram2d(
        p0_x, p0_y,
        bins=[bins_x, bins_y],
        range=[[0, img_w], [0, img_h]]
    )

    heatmap_t1, _, _ = np.histogram2d(
        p1_x, p1_y,
        bins=[bins_x, bins_y],
        range=[[0, img_w], [0, img_h]]
    )

    diff_matrix = heatmap_t1 - heatmap_t0
    return diff_matrix, bins_x, bins_y


def main():
    st.markdown("<h1 style='text-align: center;'> Military Detection & Analysis </h1>", unsafe_allow_html=True)

    model = load_sahi_model(MODEL_PATH)
    if not model:
        st.stop()

    st.sidebar.title("Options Menu")
    app_mode = st.sidebar.radio("Select Operation Mode:",
                                ["Object Detection (Single Image)", "Tactical Map (Comparative T0 vs T1)"])

    if app_mode == "Object Detection (Single Image)":
        st.header("Military Object Detection")

        uploaded_file = st.file_uploader("Upload an image (.jpg, .png):", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            st.markdown("---")

            if st.button("Detect Objects"):
                with st.spinner("Analyzing image..."):
                    image_np = np.array(image)

                    result = get_sliced_prediction(
                        image_np,
                        model,
                        slice_height=640,
                        slice_width=640,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2
                    )

                    visual_result = visualize_object_predictions(
                        image=image_np,
                        object_prediction_list=result.object_prediction_list,
                    )

                    st.image(visual_result["image"], caption="Final Result", use_container_width=True)

                    counts = {}
                    for prediction in result.object_prediction_list:
                        class_name = prediction.category.name
                        counts[class_name] = counts.get(class_name, 0) + 1

                    if counts:
                        st.success(f"Detection complete! Total objects: {len(result.object_prediction_list)}")
                        st.json(counts)
                    else:
                        st.warning("No objects detected.")

    elif app_mode == "Tactical Map (Comparative T0 vs T1)":
        st.header("Tactical Map & Density Analysis")
        st.info("Analyzes movement and calculates vehicle density per Hectare.")

        col1, col2 = st.columns(2)

        with col1:
            file1 = st.file_uploader("Image T0 (Previous)", type=['jpg', 'png', 'jpeg'], key="h1")
        with col2:
            file2 = st.file_uploader("Image T1 (Current)", type=['jpg', 'png', 'jpeg'], key="h2")

        if file1 and file2:
            img1_pil = Image.open(file1)
            img2_pil = Image.open(file2)
            if img1_pil.size != img2_pil.size:
                st.warning(
                    f"Warning: Images have different dimensions ({img1_pil.size} vs {img2_pil.size}). Results may be inaccurate.")

            if st.button("Generate Tactical Map"):
                with st.spinner("Calculating tactical movements and density..."):
                    img1_np = np.array(img1_pil)
                    img2_np = np.array(img2_pil)

                    points_t0 = coords(img1_np, model)
                    points_t1 = coords(img2_np, model)
                    img_w, img_h = img1_pil.size

                    diff_matrix, bx, by = calculate_tactical_heatmap(points_t0, points_t1, img_w, img_h)

                    totalarea_sqm = bx * by * (50 * 50)
                    totalarea_ha = totalarea_sqm / 10000

                    total_vehicles_t1 = len(points_t1)
                    avgd = 0
                    if totalarea_ha > 0:
                        avgd = total_vehicles_t1 / totalarea_ha

                    st.markdown(" Density Report (Moment T1)")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Vehicles Detected", f"{total_vehicles_t1}")
                    m2.metric("Analyzed Area", f"{totalarea_ha:.2f} ha")
                    m3.metric("Avg Density", f"{avgd:.2f} veh/ha")
                    st.markdown("---")
                    st.write(f"Tactical Grid Resolution: {bx}x{by} sectors")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        diff_matrix.T,
                        cmap="vlag",
                        center=0,
                        annot=True,
                        fmt='.0f',
                        ax=ax
                    )
                    ax.set_title("Movement Heatmap (Red = Arrival, Blue = Departure)")
                    ax.set_xlabel("X COORDINATE (Sectors)")
                    ax.set_ylabel("Y COORDINATE (Sectors)")
                    st.pyplot(fig)


if __name__ == "__main__":
    main()