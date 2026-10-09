"""Generate complete Day 23 Sensor Fusion Bonus artifacts:
1. Calibration Drift Analysis (extrinsic perturbation, innovation magnitude, chi2 rejection, RMSE impact).
2. Visualizations (BEV tracking with pointcloud/detections/tracks, Camera FRONT projected 3D bounding boxes).
3. CVAT Track Export (export_tracks_json).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "platform"))
sys.path.insert(0, str(ROOT / "platform" / "third_party" / "waymo_reader"))

from fusion_lab.paths import student_root, PLATFORM_ROOT
from fusion_lab.workspace_loader import load_workspace
from fusion_lab.tracking_params import dt
from fusion_lab.tracking.filter import Filter
from fusion_lab.tracking.manager import TrackManager
from fusion_lab.tracking.sensors import Sensor, Measurement
from fusion_lab.lidar_pcl import pcl_from_range_image
from fusion_lab.evaluation import valid_ground_truth, tracking_counts
from fusion_lab.export_cvat import export_tracks_json

from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils as waymo_utils


def get_3d_box_corners(x, y, z, h, w, l, yaw):
    """Return 8 3D corners in vehicle frame."""
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    # 8 local corners: dx, dy, dz
    dx = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    dy = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    dz = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    corners = []
    for i in range(8):
        cx = x + cos_y * dx[i] - sin_y * dy[i]
        cy = y + sin_y * dx[i] + cos_y * dy[i]
        cz = z + dz[i]
        corners.append([cx, cy, cz])
    return np.array(corners)


def project_points_to_cam(points_veh, sensor, cam_mod):
    """Project 3D vehicle points to camera pixel (u, v)."""
    pixels = []
    for p in points_veh:
        state = np.asmatrix(np.r_[p, [0, 0, 0]]).T
        if sensor.in_fov(state):
            try:
                pred = cam_mod.camera_measurement_prediction(state, sensor)
                pixels.append((float(pred[0, 0]), float(pred[1, 0])))
            except ValueError:
                pixels.append(None)
        else:
            pixels.append(None)
    return pixels


def run_bonus_generation():
    bonus_dir = student_root() / "bonus"
    bonus_dir.mkdir(parents=True, exist_ok=True)

    paths_file = student_root() / "config" / "paths.yaml"
    with paths_file.open() as f:
        paths_cfg = yaml.safe_load(f)

    tfrecord_path = (student_root() / paths_cfg["waymo_dir"] / paths_cfg["segment"]).resolve()
    weights_path = (student_root() / paths_cfg["weights_dir"] / "pretrained_fpn-resnet" / "fpn_resnet_18_epoch_300.pth").resolve()

    ws = load_workspace()
    kalman = ws["kalman"]
    assoc = ws["association"]
    cam = ws["camera_fusion"]
    bev = ws["bev_mapping"]
    det_pipe = ws["detection_pipeline"]
    det_metrics = ws["detection_metrics"]

    det_cfg = det_pipe.load_fpn_resnet_config(str(weights_path))
    model = det_pipe.create_fpn_model(det_cfg, str(weights_path))

    print("Step 1: Running Calibration Drift Experiments...")
    drift_levels = [
        {"name": "0. Baseline (No drift)", "delta_y": 0.0, "delta_yaw": 0.0},
        {"name": "1. Slight Drift (dy=0.2m, dyaw=0.02 rad)", "delta_y": 0.2, "delta_yaw": 0.02},
        {"name": "2. Moderate Drift (dy=0.6m, dyaw=0.06 rad)", "delta_y": 0.6, "delta_yaw": 0.06},
        {"name": "3. Severe Drift (dy=1.5m, dyaw=0.15 rad)", "delta_y": 1.5, "delta_yaw": 0.15},
    ]

    drift_results = []
    test_frames_count = 60  # Evaluate over 60 frames for quick, reliable comparison

    for drift in drift_levels:
        reader = WaymoDataFileReader(str(tfrecord_path))
        KF = Filter(kalman)
        manager = TrackManager(ws["track_management"])
        lidar_sensor = None
        camera_sensor = None
        rng = np.random.default_rng(0)

        total_sq_err = 0.0
        total_matches = 0
        total_cam_attempted = 0
        total_cam_accepted = 0
        innovations = []

        for cnt, frame in enumerate(reader):
            if cnt >= test_frames_count:
                break
            if lidar_sensor is None:
                lidar_calib = waymo_utils.get(frame.context.laser_calibrations, dataset_pb2.LaserName.TOP)
                lidar_sensor = Sensor("lidar", lidar_calib, cam)
            if camera_sensor is None:
                cam_calib = waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
                camera_sensor = Sensor("camera", cam_calib, cam)
                # Apply extrinsic drift perturbation
                trans = np.asarray(camera_sensor.veh_to_sens).copy()
                trans[1, 3] += drift["delta_y"]
                # Rotate yaw around z-axis in sensor frame
                theta = drift["delta_yaw"]
                R_z = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                trans[:3, :3] = R_z @ trans[:3, :3]
                camera_sensor.veh_to_sens = np.asmatrix(trans)
                camera_sensor.sens_to_veh = np.asmatrix(np.linalg.inv(trans))

            points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
            tensor = torch.from_numpy(bev.bev_maps_from_pcl(points, det_cfg)).unsqueeze(0).float()
            detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)
            labels = valid_ground_truth(frame.laser_labels, det_cfg, label_pb2.Label.Type.TYPE_VEHICLE)

            # Lidar pass
            observations = []
            for detection in detections:
                if (det_cfg.lim_x[0] <= detection[1] <= det_cfg.lim_x[1] and
                    det_cfg.lim_y[0] <= detection[2] <= det_cfg.lim_y[1]):
                    lidar_sensor.generate_measurement(cnt, detection[1:], observations)

            for track in manager.track_list:
                KF.predict(track)
                track.set_t(cnt * dt)

            assoc.associate_and_update(manager, observations, KF, lidar_sensor)

            # Camera pass
            group = next((g for g in frame.camera_labels if g.name == dataset_pb2.CameraName.FRONT), None)
            if group is not None:
                cam_observations = []
                for label in group.labels:
                    if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                        centre = np.array([label.box.center_x, label.box.center_y])
                        camera_sensor.generate_measurement(cnt, centre + rng.normal(0, 0.5, 2), cam_observations)

                # Track camera acceptance rate and innovation
                cost_mat = assoc.association_cost_matrix(manager.track_list, cam_observations)
                for trk in manager.track_list:
                    for obs in cam_observations:
                        if camera_sensor.in_fov(trk.x):
                            total_cam_attempted += 1
                            gamma = kalman.innovation(trk.x, obs)
                            inno_mag = float(np.linalg.norm(gamma))
                            innovations.append(inno_mag)
                            d2 = assoc.mahalanobis_distance(trk, obs)
                            if assoc.chi2_gate(d2, camera_sensor):
                                total_cam_accepted += 1

                assoc.associate_and_update(manager, cam_observations, KF, camera_sensor)

            t_counts = tracking_counts(manager.track_list, labels)
            total_matches += t_counts["matches"]
            total_sq_err += t_counts["sum_sq_err"]

        rmse = np.sqrt(total_sq_err / total_matches) if total_matches > 0 else 0.0
        acc_rate = (total_cam_accepted / total_cam_attempted * 100.0) if total_cam_attempted > 0 else 0.0
        mean_inno = np.mean(innovations) if innovations else 0.0

        drift_results.append({
            "level": drift["name"],
            "delta_y_m": drift["delta_y"],
            "delta_yaw_rad": drift["delta_yaw"],
            "rmse_m": float(rmse),
            "matches": total_matches,
            "mean_innovation_px": float(mean_inno),
            "gate_acceptance_pct": float(acc_rate),
        })

    # Save drift results JSON
    (bonus_dir / "calibration_drift_analysis.json").write_text(json.dumps(drift_results, indent=2))
    print("Calibration Drift Results:")
    for r in drift_results:
        print(f"  {r['level']}: RMSE={r['rmse_m']:.4f}m, Mean Innov={r['mean_innovation_px']:.1f}px, Gate Accept={r['gate_acceptance_pct']:.1f}%")

    print("\nStep 2: Generating Visualizations (BEV and Camera Front)...")
    # Generate 2 rich visualization figures
    reader = WaymoDataFileReader(str(tfrecord_path))
    data_iter = iter(reader)

    manager = TrackManager(ws["track_management"])
    KF = Filter(kalman)
    lidar_sensor = None
    camera_sensor = None
    rng = np.random.default_rng(0)

    cvat_frames = []

    for cnt, frame in enumerate(data_iter):
        if cnt > 15:
            break
        if lidar_sensor is None:
            lidar_sensor = Sensor("lidar", waymo_utils.get(frame.context.laser_calibrations, dataset_pb2.LaserName.TOP), cam)
            camera_sensor = Sensor("camera", waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT), cam)

        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        tensor = torch.from_numpy(bev.bev_maps_from_pcl(points, det_cfg)).unsqueeze(0).float()
        detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)
        labels = valid_ground_truth(frame.laser_labels, det_cfg, label_pb2.Label.Type.TYPE_VEHICLE)

        observations = []
        for d in detections:
            if det_cfg.lim_x[0] <= d[1] <= det_cfg.lim_x[1] and det_cfg.lim_y[0] <= d[2] <= det_cfg.lim_y[1]:
                lidar_sensor.generate_measurement(cnt, d[1:], observations)

        for track in manager.track_list:
            KF.predict(track)
            track.set_t(cnt * dt)

        assoc.associate_and_update(manager, observations, KF, lidar_sensor)

        # Camera pass
        group = next((g for g in frame.camera_labels if g.name == dataset_pb2.CameraName.FRONT), None)
        cam_obs = []
        if group is not None:
            for label in group.labels:
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    centre = np.array([label.box.center_x, label.box.center_y])
                    camera_sensor.generate_measurement(cnt, centre + rng.normal(0, 0.5, 2), cam_obs)
            assoc.associate_and_update(manager, cam_obs, KF, camera_sensor)

        # Record for CVAT
        frame_tracks = []
        for trk in manager.track_list:
            if trk.state == "confirmed":
                frame_tracks.append({
                    "id": trk.id,
                    "x": float(trk.x[0, 0]),
                    "y": float(trk.x[1, 0]),
                    "z": float(trk.x[2, 0]),
                    "vx": float(trk.x[3, 0]),
                    "vy": float(trk.x[4, 0]),
                    "vz": float(trk.x[5, 0]),
                    "h": float(trk.height),
                    "w": float(trk.width),
                    "l": float(trk.length),
                    "yaw": float(trk.yaw),
                })
        cvat_frames.append({"frame": cnt, "tracks": frame_tracks})

        # Render visualizations for frame 10
        if cnt == 10:
            # 1. BEV Visualization Figure
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            bev_map = bev.bev_maps_from_pcl(points, det_cfg)
            density = bev_map[2]
            ax.imshow(density, cmap="inferno", extent=[det_cfg.lim_y[0], det_cfg.lim_y[1], det_cfg.lim_x[0], det_cfg.lim_x[1]], origin="lower")

            # Plot Detections
            for d in detections:
                ax.plot(d[2], d[1], "go", markersize=8, label="LiDAR Detection (FPN)" if d is detections[0] else "")

            # Plot Confirmed Tracks
            for trk in manager.track_list:
                color = "cyan" if trk.state == "confirmed" else "orange"
                ax.plot(trk.x[1, 0], trk.x[0, 0], "s", color=color, markersize=12, label=f"Track ID {trk.id} ({trk.state})")
                ax.text(trk.x[1, 0] + 0.8, trk.x[0, 0], f"ID:{trk.id} (v={trk.x[3,0]:.1f}m/s)", color="white", fontsize=10, weight="bold")

            # Plot Ground Truth
            for lbl in labels:
                ax.plot(lbl.box.center_y, lbl.box.center_x, "rx", markersize=10, markeredgewidth=2, label="Ground Truth" if lbl is labels[0] else "")

            ax.set_title(f"BEV Multi-Sensor Tracking (Frame {cnt}) - Tracks vs Detections vs GT", fontsize=14, weight="bold")
            ax.set_xlabel("Lateral Y (m)", fontsize=12)
            ax.set_ylabel("Longitudinal X (m)", fontsize=12)
            ax.set_xlim(det_cfg.lim_y)
            ax.set_ylim(det_cfg.lim_x)
            ax.grid(True, linestyle="--", alpha=0.5)
            # Remove duplicate labels in legend
            handles, labels_leg = ax.get_legend_handles_labels()
            by_label = dict(zip(labels_leg, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")
            fig.tight_layout()
            fig.savefig(str(bonus_dir / "viz_bev_tracking_frame_0010.png"))
            plt.close(fig)

            # 2. Camera Front Image with Projected 3D Bounding Boxes
            front_img_data = next((im for im in frame.images if im.name == dataset_pb2.CameraName.FRONT), None)
            if front_img_data:
                img_arr = cv2.imdecode(np.frombuffer(front_img_data.image, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                h_img, w_img, _ = img_rgb.shape

                fig_cam, ax_cam = plt.subplots(figsize=(12, 7), dpi=150)
                ax_cam.imshow(img_rgb)

                for trk in manager.track_list:
                    if trk.state == "confirmed":
                        corners = get_3d_box_corners(
                            float(trk.x[0, 0]), float(trk.x[1, 0]), float(trk.x[2, 0]),
                            float(trk.height), float(trk.width), float(trk.length), float(trk.yaw)
                        )
                        proj = project_points_to_cam(corners, camera_sensor, cam)
                        if all(p is not None for p in proj):
                            # Draw bottom rectangle (corners 4,5,6,7) and top rectangle (0,1,2,3)
                            lines = [
                                (0, 1), (1, 2), (2, 3), (3, 0), # top
                                (4, 5), (5, 6), (6, 7), (7, 4), # bottom
                                (0, 4), (1, 5), (2, 6), (3, 7)  # pillars
                            ]
                            for p1, p2 in lines:
                                ax_cam.plot([proj[p1][0], proj[p2][0]], [proj[p1][1], proj[p2][1]], color="cyan", linewidth=2)

                            top_mid = proj[0]
                            ax_cam.text(top_mid[0], top_mid[1] - 15, f"Track {trk.id} [Fused: x={trk.x[0,0]:.1f}m, y={trk.x[1,0]:.1f}m]",
                                        color="yellow", fontsize=10, weight="bold",
                                        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2))

                ax_cam.set_title(f"Camera FRONT Projection with 3D Track Bounding Boxes (Frame {cnt})", fontsize=14, weight="bold")
                ax_cam.axis("off")
                fig_cam.tight_layout()
                fig_cam.savefig(str(bonus_dir / "viz_camera_front_projection_frame_0010.png"))
                plt.close(fig_cam)

    print("Step 3: Exporting CVAT Tracks JSON...")
    export_tracks_json(cvat_frames, bonus_dir / "cvat_tracks_export.json")

    print("\nAll Bonus artifacts generated successfully in student/bonus/!")


if __name__ == "__main__":
    run_bonus_generation()

