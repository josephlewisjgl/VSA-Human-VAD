import os, cv2, numpy as np, pandas as pd
from tqdm import tqdm
import torch
from .sortalg import Sort




def estimate_train_vids(video_dir, model, FEATURE_EXTRACTOR, plot=False):
    '''
    Based on a directory of video frames, infer poses and assign IDs to create 
    a tabular dataset of key points (joints) and their position

    :param video_dir: Directory of videos to inference on
    '''
    
    results = []

    with torch.no_grad():
        for video_name in os.listdir(video_dir):
            
            # account for .avi files in dir
            if "_frames" not in video_name:
                continue

            print(f"Processing video: {video_name}")
            frame_dir = os.path.join(video_dir, video_name)
            if not os.path.isdir(frame_dir):
                continue

            # directory to save pose plots alongside frames
            if plot:
                pose_plot_dir = os.path.join(frame_dir, "pose_plots")
                os.makedirs(pose_plot_dir, exist_ok=True)
            
            # initialise a new tracker at each video (prevent mistaken linking)
            tracker = Sort(max_age=10)
            frame_files = sorted(os.listdir(frame_dir))

            # process each file individually
            for frame_idx, frame_file in tqdm(enumerate(frame_files), total=len(frame_files)):
                # read the image
                frame_path = os.path.join(frame_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                do_plot = plot & (frame_idx%10 == 0)
                
                # make predictions
                # lower conf for quantised model to allow more detections
                if FEATURE_EXTRACTOR == 'edge':
                    pred = model.predict(source=frame, conf=0.15, iou=0.3, imgsz=(160,320), save=False, verbose=False)
                else:
                    pred = model.predict(source=frame, save=False, verbose=False)

                # check that some predictions were made
                p0 = pred[0]
                if p0.keypoints is None or len(p0.boxes) == 0:
                    tracker.update(np.empty((0, 5)))
                    continue
                    
                # get the bounding boxes, confidence scores (for track) and keypoints
                boxes = p0.boxes.xyxy.numpy()
                scores = p0.boxes.conf.numpy()
                keypoints = p0.keypoints.data.numpy()

                # add the detections to the tracker (based on bounding box)
                detections = np.hstack([boxes, scores[:, None]])
                tracked = tracker.update(detections)

                # for matching indviduals
                centroids = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2,
                                            (boxes[:, 1] + boxes[:, 3]) / 2))

                # for each bounding box track identify the pose for this ID
                for x1, y1, x2, y2, track_id in tracked:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if len(centroids) > 1:
                        matched_idx = np.argmin(np.linalg.norm(centroids - (cx, cy), axis=1))
                    else:
                        matched_idx = 0

                    # pull out the key points
                    kp = keypoints[matched_idx, :, :2].flatten()

                    row = {
                        "video": video_name,
                        "frameID": frame_idx,
                        "personID": int(track_id),
                        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2
                    }
                    
                    # append each key point column wise
                    for j in range(17):
                        row[f"joint{j+1}x"] = kp[2 * j]
                        row[f"joint{j+1}y"] = kp[2 * j + 1]
                    results.append(row)

                    if do_plot:
                        plot_img = frame.copy()
                        _draw_points_and_id(plot_img, keypoints[matched_idx, :, :2], (x1, y1, x2, y2), track_id)

                if do_plot and (tracked.size > 0):
                    # Save plot image next to frames under 'pose_plots'
                    out_name = f"{os.path.splitext(frame_file)[0]}_pose_labelled.jpg"
                    out_path = os.path.join(pose_plot_dir, out_name)
                    cv2.imwrite(out_path, plot_img)

            print(f"Processed {video_name} with {len(results)} detections so far.")

        if 'train' in video_dir:
            filename = "train_tracked_poses_n.csv" if FEATURE_EXTRACTOR == 'edge' else "train_tracked_poses_m.csv"
        else:
            filename = "test_tracked_poses_n.csv" if FEATURE_EXTRACTOR == 'edge' else "test_tracked_poses_m.csv"
        pd.DataFrame(results).to_csv('./data/' + filename, index=False)


def _color_for_id(track_id):
    # Stable pseudo-random color per ID (BGR for OpenCV)
    rng = np.random.RandomState(int(track_id) * 97 % 2**32)
    return tuple(int(x) for x in rng.randint(64, 255, size=3))  # avoid very dark colors

def _draw_points_and_id(img, keypoints_xy, bbox, track_id):
    color = _color_for_id(track_id)
    kp = keypoints_xy.astype(int)

    # draw joints only (no skeleton)
    for x, y in kp:
        cv2.circle(img, (int(x), int(y)), 3, color, -1, lineType=cv2.LINE_AA)

    # bbox and label for context
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    label = f"ID {int(track_id)}"
    cv2.putText(img, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)