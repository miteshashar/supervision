import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


def main(
    source_video_path: str,
    zone_configuration_path: str,
    weights: str,
    device: str,
    confidence: float,
    iou: float,
    classes: List[int],
) -> None:
    model = YOLO(weights)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]
    time_stream_path = Path(
        source_video_path.replace(
            ".mp4", f"_{type(video_info.fps).__name__}_time_stream.csv"
        )
    )
    time_stream_writer = csv.writer(time_stream_path.open("w", encoding="utf-8"))
    time_stream_writer.writerow(["frame_number", "zone_id", "tracker_id", "time"])
    with sv.VideoSink(
        source_video_path.replace(".mp4", f"_{type(video_info.fps).__name__}.mp4"),
        video_info,
    ) as sink:
        current_time_stream = []
        for frame_number, frame in enumerate(frames_generator):
            results = model(frame, conf=confidence)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[find_in_list(detections.class_id, classes)]
            detections = detections.with_nms(threshold=iou)
            detections = tracker.update_with_detections(detections)

            annotated_frame = frame.copy()

            for idx, zone in enumerate(zones):
                annotated_frame = sv.draw_polygon(
                    scene=annotated_frame,
                    polygon=zone.polygon,
                    color=COLORS.by_idx(idx),
                )

                detections_in_zone = detections[zone.trigger(detections)]
                time_in_zone = timers[idx].tick(detections_in_zone)
                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                annotated_frame = COLOR_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup,
                )
                labels = [
                    f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                    for tracker_id, time in zip(
                        detections_in_zone.tracker_id, time_in_zone
                    )
                ]
                annotated_frame = LABEL_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    labels=labels,
                    custom_color_lookup=custom_color_lookup,
                )
                current_time_stream.append(
                    {
                        str(tracker_id): time
                        for tracker_id, time in zip(detections.tracker_id, time_in_zone)
                    }
                )
                for tracker_id, time in zip(detections.tracker_id, time_in_zone):
                    time_stream_writer.writerow(
                        [frame_number, idx, tracker_id, round(time, 6)]
                    )

            sink.write_frame(annotated_frame)
            # time_stream.append(current_time_stream)

            # cv2.imshow("Processed Video", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
        # cv2.destroyAllWindows()

        # time_stream_path.write_text(json.dumps(time_stream, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using video file."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8s.pt",
        help="Path to the model weights file. Default is 'yolov8s.pt'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu', 'mps' or 'cuda'). Default is 'cpu'.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        weights=args.weights,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )
