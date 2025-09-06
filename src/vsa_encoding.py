import numpy as np
import pandas as pd

import torchhd as hd
import torch

from tqdm import tqdm

from sklearn.cluster import KMeans

def calc_centroid(df):
    '''
    Compute the centroid of a person based on their joints

    :param df: Dataframe to compute for
    :returns: pd.DataFrame Data frame with centroidx and centroid y added
    '''

    # use all joints
    joint_columns = [f'joint{i}{axis}' for i in range(1, 18) for axis in ['x', 'y']]

    df['centroidx'] = df[[col for col in joint_columns if col.endswith('x')]].mean(axis=1)
    df['centroidy'] = df[[col for col in joint_columns if col.endswith('y')]].mean(axis=1)

    return df


def calc_area(df):
    '''
    Calculates the area of a bounding box around a person

    :param df: Df with bboxx1, bboxx2, bboxy1, bboxy2
    :returns: pd.DataFrame df with bounding box area (bbox_area) added
    '''
    
    df['bbox_area'] = (df['bbox_x2'] - df['bbox_x1']) * (df['bbox_y2'] - df['bbox_y1'])
    return df

def get_aspect_thresh(detections, nbins=3):
    """
    Get a dictionary of bin edges based on detections and nbins

    If nbins == 3 use standard deviation based bins else use 5 percentile based bins (ablation)

    :param detections: Dataframe of detections with bbox_area 
    :param nbins: Number of bins (3/5)
    :returns: dict Bin edges and labels
    """
    if nbins not in (3, 5):
        raise ValueError("nbins must be either 3 or 5")

    label_map = {
        3: ["far", "middle", "close"],
        5: ["very far", "far", "middle", "close", "very close"],
    }

    # only use valid bbox areas
    col_vals = detections["bbox_area"].to_numpy()
    col_vals = col_vals[np.isfinite(col_vals)]

    if nbins == 3:
        mean = float(np.mean(col_vals))
        std = float(np.std(col_vals))
        low = mean -  std
        high = mean + std

        # cases where aspect is constant (one/no detections)
        if high <= low:
            high = np.nextafter(low, np.inf)

        bins = np.array([0.0, low, high, np.inf], dtype=float)

    # five bins use percentiles
    else:
        mean = float(np.mean(col_vals))
        std = float(np.std(col_vals))
        very_low = mean - 2 * std if mean - 2 * std > 0 else 0.0001
        low = mean - std
        high = mean + std
        very_high = mean + 2 * std

        bins = np.array([0.0, very_low, low, high, very_high, np.inf], dtype=float)

        for b in range(len(bins)-1):
            if bins[b+1] == bins[b]:
                bins[b+1] = np.nextafter(bins[b], np.inf) 

    # map of label names to bin size for symbol encoding
    return {"bins": bins.tolist(), "labels": label_map[nbins]}

def bin_aspect(detections, aspect_thresh):
    """
    Apply binning to 'bbox_area' using labels computed earlier

    :param detections: Dataframe with detections
    :param aspect_thresh: Dictionary of thresholds
    :returns: pd.DataFrame detections with aspectbin col added
    """
    df = detections.copy()
    df["aspectbin"] = pd.cut(
        df["bbox_area"],
        bins=aspect_thresh["bins"],
        labels=aspect_thresh["labels"],
        right=False,
        include_lowest=True,
    )
    return df


def identify_grid_pos(detections, img_width=640, img_height=360,
                      grid_size_x=5, grid_size_y=5):
    '''
    Add a position col to a dataframe

    :param detections: Detections data
    :param img_width: Width of the image
    :param img_height: Height of the image
    :param grid_size_x: Desired grid size
    :param grid_size_y: Desired grid size
    :returns: pd.DataFrame detections with position col added
    '''

    # identify the grid cell (rounded pos/width or pos/height)
    x_bin = (detections['centroidx'] / img_width * grid_size_x).astype(int)
    y_bin = (detections['centroidy'] / img_height * grid_size_y).astype(int)

    # convert to single 1D label (y_bin * gridsize moves down + xbin moves across)
    detections['position'] = y_bin * grid_size_x + x_bin + 1
    return detections


def add_time_in_scene(df):
    '''
    Adds a time in scene variable to detections

    :param df: Dataframe containing detections
    :returns: pd.DataFrame with time field
    '''

    # ensure time calculated per person/frame/video combo starting from first frame
    df = df.sort_values(by=['video', 'personID', 'frameID']).reset_index(drop=True)


    # count the time in each segment (+1 for 0 frame indexing)
    df['time_in_scene'] = df.groupby(['video', 'personID']).cumcount() + 1

    return df


def get_time_thresh(detections, nbins = 3):
    """
    Compute percentile only thresholds for time feature

    :param detections: Dataframe of detections
    :param nbins: Number of bins
    """

    # map for labels
    label_map = {
        3: ["short", "normal", "long"],
        5: ["very short", "short", "normal", "long", "very long"]
    }

    # evenly spaced bins along the percentiles either at 3rds or 5ths
    percentiles = np.linspace(0, 100, nbins + 1)
    bins = np.percentile(detections["time_in_scene"], percentiles)
    
    # ensure that the last bin is open and first bin 0 (no one can have negative time in frame)
    bins[0] = 0  
    bins[-1] = np.inf 

    # edges must all increase
    for i in range(1, len(bins)-1):
        if bins[i] <= bins[i-1]:
            bins[i] = np.nextafter(bins[i-1], np.inf)

    return {"bins": bins.tolist(), "labels": label_map[nbins]}

def bin_time(detections, time_thresh):
    """
    Apply time bins to detections

    :param detections: Dataframe containing detections
    :param time_thresh: Pre calculated time thresholds with labels
    :returns: Detections with time threshold bins in timebin col
    """
    df = detections.copy()
    df["timebin"] = pd.cut(
        df["time_in_scene"],
        bins=time_thresh["bins"],
        labels=time_thresh["labels"],
        right=False,
        include_lowest=True
    )
    return df

def angle(a, b, c):
    '''
    Compute the angle between 3 points

    :param a: First point
    :param b: Mid point
    :param c: Third point
    :returns: Angle
    '''
    
    # compute vectors a -> b c -> b
    ba = a - b
    bc = c - b

    # dot product/vector length
    cos_theta = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))

    # np.arccos returns radians np.degrees converts to degrees (clamp for floating point inaccuracy)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def classify_leg_posture(df):
    '''
    Classify leg posture into pose estimation labels

    Fill unknown then:
    if knees bent and torso straight then sitting
    if knees bent and torso slouched then crouching
    if knees straight and torso slouched then leaning
    if knees straight and torso straight then standing

    :params df: Dataframe of pose estimations
    :returns: pd.DataFrame of detections with leg_posture added
    '''
    
    # isolate pose keypoints for leg angle
    l_hip = df[['joint12x', 'joint12y']].to_numpy()
    l_knee = df[['joint14x', 'joint14y']].to_numpy()
    l_ankle = df[['joint16x', 'joint16y']].to_numpy()
    r_hip = df[['joint13x', 'joint13y']].to_numpy()
    r_knee = df[['joint15x', 'joint15y']].to_numpy()
    r_ankle = df[['joint17x', 'joint17y']].to_numpy()
    l_sh = df[['joint6x', 'joint6y']].to_numpy()
    r_sh = df[['joint7x', 'joint7y']].to_numpy()

    # compute angle hip -> knee -> ankle
    l_angle = angle(l_hip, l_knee, l_ankle)
    r_angle = angle(r_hip, r_knee, r_ankle)
    knee_angle = np.nanmax([l_angle, r_angle], axis=0)

    # shoulder and hip average positions 
    shoulder_mid = (l_sh + r_sh) / 2 
    hip_mid = (l_hip + r_hip) / 2     
    
    # hypothetical straight line through shoulders (straight down)
    vertical_point = shoulder_mid.copy()
    vertical_point[:, 1] += 100
    
    # hip -> shoulder -> straight line for angle
    torso_tilt = angle(hip_mid, shoulder_mid, vertical_point)

    # compute poses 
    labels = np.full(len(df), 'unknown', dtype=object)
    labels[(knee_angle < 90) & (torso_tilt <= 15)] = 'sitting'
    labels[(knee_angle < 90) & (torso_tilt > 15)] = 'crouching'
    labels[(knee_angle > 160) & (torso_tilt > 15)] = 'leaning'
    labels[(knee_angle > 160) & (torso_tilt <= 15)] = 'standing'

    df['leg_posture'] = labels

    return df


def build_encoding_df(df, aspect_thresh=None, time_thresh=None, bins=3):
    ''''
    Process df using pre-defined funcs

    :param df: Dataframe of detections
    :param aspect_thresh: If this is the test dataset don't recompute thresholds
    :param time_thresh: As above
    :param bins: How many bins to use
    :returns: pd.DataFrame with processed cols added
    '''

    # add all cols 
    df = calc_centroid(df)
    df = calc_area(df)
    df = add_time_in_scene(df)
    df = identify_grid_pos(df, grid_size_x=bins, grid_size_y=bins)
    df = classify_leg_posture(df)

    # bin the aspect and time cols using predefined/computed thresholds
    if aspect_thresh is None:
      aspect_thresh = get_aspect_thresh(df, nbins=bins)

    df = bin_aspect(df, aspect_thresh)

    if time_thresh is None:
      time_thresh = get_time_thresh(df, nbins=bins)

    df = bin_time(df, time_thresh)

    # reset index due to sort ops + returns threshes to use in test (if train)
    return df.reset_index(drop=True), aspect_thresh, time_thresh


def encode_vsa(df, Features, Pos, Aspect, Times, Postures, bins=3):
    '''
    Build VSA encoding from detections df

    Takes the VSA basis vectors as test must use same as train

    Features[0] = Time
    Features[1] = Position
    Features[2] = Aspect/Range
    Features[3] = Posture

    :param df: Dataframe of processed detections
    :param Features: Feature basis vecs
    :param Pos: Grid position basis vecs
    :param Aspect: Aspect/bbox area basis vecs
    :param Times: Time basis vecs
    :param Postures: Posture basis vecs
    :param bins: Number of bins used in system
    '''

    # for ablation to five bins
    if bins == 3:
        aspect_labels = ['far', 'middle', 'close']
        time_labels = ['short', 'normal', 'long']
    elif bins == 5:
        aspect_labels = ['very far', 'far', 'middle', 'close', 'very close']
        time_labels = ['very short', 'short', 'normal', 'long', 'very long']

    # static (no ablation)
    posture = ['crouching', 'standing', 'leaning', 'unknown', 'sitting']

    # to hold computed vecs
    person_vecs = []

    # for each row (as tuple for efficiency)
    for r in tqdm(df.itertuples(index=False), total=len(df)):
        # identify a feature to use and the appropriate feature vector and bind
        vec = hd.bind(Features[0], Times[time_labels.index(r.timebin)])

        # then repeat for every feature bundling into the original
        vec = hd.bundle(vec, hd.bind(Features[1], Pos[min((bins*bins)-1,r.position)]))
        vec = hd.bundle(vec, hd.bind(Features[2], Aspect[aspect_labels.index(r.aspectbin)]))
        vec = hd.bundle(vec, hd.bind(Features[3], Postures[posture.index(r.leg_posture)]))
        person_vecs.append(torch.sign(vec))


    return person_vecs


def build_temporal_enc(detections, person_vecs, step=5, stride=25):
    '''
    Temporal encoding (rotate and bundle)

    :param detections: Original detections for sorting frames
    :param person_vecs: VSA person vector list
    :param step: Number of vectors to bundle
    :param stride: Distance between vecs to bundle
    :returns: List of vectors with temporal bundling applied
    '''
    
    # build dict of keys: video, personID, values: frameID, index
    unique_vid_person = detections.groupby(['video', 'personID']).frameID.size().reset_index()
    person_frame_map = {(r['video'], r['personID']):[] for i, r in unique_vid_person.iterrows()}

    for idx, row in detections.iterrows():
        key = (row['video'], row['personID'])
        person_frame_map[key].append((row['frameID'], idx))

    # sort frame id within person to retain order
    for key in person_frame_map:
        person_frame_map[key].sort()

    temporal_vecs = []

    # for each detection
    for _, row in tqdm(detections.iterrows()):
        # get the vid, personid, frameid combo
        vid, pid, fid = row['video'], row['personID'], row['frameID']
        key = (vid, pid) # isolate just the dict key

        # get the past frames for the person
        past_frames = person_frame_map[key]

        # find where this frame is and it's position in the past frames collection
        current_idx_in_past = next(
            (j for j, (f, _) in enumerate(past_frames) if f == fid),
            None
        )

        # initialise empty vec
        vec = torch.zeros_like(person_vecs[0]) 

        # use 25 FPS logic so the step is 25 stride is 5
        for offset in range(step):
            # current position - 25 breaking if not exists
            history_idx = current_idx_in_past - offset*stride
            if history_idx < 0:
                break
            
            # pull out the vector 25 frames earlier
            _, vec_index_j = past_frames[history_idx]
            vec_j = person_vecs[vec_index_j]

            # then perumte it by number of steps and bundle
            vec = hd.bundle(vec, hd.permute(vec_j, shifts=offset))

        # ensure after these operations that the bipolar vec is retained
        vec = torch.sign(vec)
        temporal_vecs.append(vec)

    return temporal_vecs


def cluster(person_vecs, n_clusters=5):
    '''
    Cluster person vectors

    :param person_vecs: Person vectors (before or after temporal bundling)
    :param n_clusters: Num clusters to assign
    :returns: list of prototypes and kmeans obj or None
    '''

    # for the edge system just 1 hypervec
    if n_clusters == 1:
        # use the first hypervec as basis
        first_vec = person_vecs[0] 
        for vec in person_vecs[1:]:
            # bundle in all other vecs
            first_vec = hd.bundle(first_vec, vec)
        # ensure sign retained
        prototype = torch.sign(first_vec)  
        return  torch.stack([prototype]), None # Single prototype, no clustering
    
    # fit KMeans model with fixed seed
    X_np = person_vecs.float().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X_np)


    prototype_vectors = []

    # identify the clusters
    for cluster_id in range(n_clusters):
        cluster_vecs = person_vecs[kmeans.labels_ == cluster_id]

        # bundle all clustered vectors starting from first
        first_vec = cluster_vecs[0]
        for vec in cluster_vecs[1:]:
            first_vec = hd.bundle(first_vec, vec)

        # ensures bipolar
        ternary_vec = torch.sign(first_vec)
        prototype_vectors.append(ternary_vec)

    # stack (k, N. vectors)
    prototype_vectors = torch.stack(prototype_vectors)

    return prototype_vectors, kmeans


def compute_cluster_thresholds(prototypes, train_vectors, train_labels):
    '''
    Compute thresholds for classification based on clusters

    :param prototypes: Prototypes from KMeans or single prototype for edge
    :param train_vectors: Training vectors (thresholds not computed for test)
    :param train_labels: Cluster labels for training vectors
    :param percentile: Cut off for normality 
    :returns: List of thresholds
    '''
    
    # for each cluster
    num_clusters = prototypes.shape[0]
    thresholds = []

    for k in range(num_clusters):
        # identify the cluster prototype
        cluster_vecs = train_vectors[train_labels == k]
        proto = prototypes[k]

        # calculate similarity and convert to numpy
        sims = hd.cosine_similarity(proto, cluster_vecs).cpu().numpy()

        # then calculate the ODHD style threshold per cluster
        threshold = sims.mean() - (2 * sims.std())
        thresholds.append(threshold)

    return thresholds


def fine_tune_prototypes(prototypes, person_vecs, labels, epochs=3):
    """
    Use ODHD style finetuning

    :param prototypes: Prototype vecs
    :param person_vecs: Temporal or person vectors
    :param labels: Cluster labels
    :param epochs: Num epochs to fine tune
    :returns: tensor of prototypes
    """

    # prototypes to update
    updated_prototypes = prototypes.clone()
    cluster_members = {id: [] for id in labels}

    # collect all vectors for a cluster
    for i, cluster_id in enumerate(labels):
        cluster_members[cluster_id].append(person_vecs[i])

    # begin finetuning
    for e in range(epochs):
        # within each cluster
        for cluster_id in range(len(updated_prototypes)):

            # stack all members and identify the current prototype
            members = torch.stack(cluster_members[cluster_id])
            prototype = updated_prototypes[cluster_id]

            # similarity to that prototype
            sims = hd.cosine_similarity(prototype, members)

            # threshold as in the ODHD paper (see report for reference)
            threshold = sims.mean() + 2 * sims.std()

            # if vector outside of threshold bundle it to the threshold
            for i, sim in enumerate(sims):
                if sim < threshold:
                    prototype = hd.bundle(prototype, members[i])

            # as in other funcs ensure bipolar
            prototype = torch.sign(prototype)
            updated_prototypes[cluster_id] = prototype

    return updated_prototypes


def evaluate_test_vectors(prototypes, thresholds, test_vectors):
    '''
    Evaluate vectors according to prototype 

    :param prototypes: Prototypes to compare to
    :param thresholds: Classification thresholds
    :param test_vectors: Vectors to evaluate
    :returns: tuple of lists score, classification, threshold (for later processing)
    '''
    
    scores = []

    for vec in test_vectors:
        # compute similarity to all prototypes
        sims = hd.cosine_similarity(vec, prototypes)

        # most similar prototype used for classification/final score
        best_idx = torch.argmax(sims)
        best_sim = sims[best_idx]
        threshold = thresholds[best_idx]

        # perform classification
        anomaly = best_sim < threshold

        scores.append([best_sim.item(), anomaly, threshold])

    return [s[0] for s in scores], [s[1] for s in scores], [s[2] for s in scores]

