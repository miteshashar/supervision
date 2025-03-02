import csv
import json
from pathlib import Path

int_file = Path("vehicles.int.json")
float_file = Path("vehicles.float.json")

int_data = json.loads(int_file.read_text())
float_data = json.loads(float_file.read_text())

mismatches = []
# table = rich.table.Table(title="Object Speeds", show_lines=True)
# table.add_column("Frame")
# table.add_column("Object")
# table.add_column("Int Speed")
# table.add_column("Float Speed")

report_writer = csv.writer(
    open(
        str(int_file.with_suffix(".csv").absolute()).replace(".int", ".report"),
        "w",
        encoding="utf-8",
    )
)
report_writer.writerow(
    ["frame_number", "object_tracking_id", "int_speed", "float_speed", "difference"]
)

for idx, [int_frame, float_frame] in enumerate(zip(int_data, float_data)):
    current_mismatches = {
        "frame": idx + 1,
        "mismatches": [],
    }

    object_ids = sorted(list(set(int_frame.keys()) | set(float_frame.keys())))

    for object_id in object_ids:
        int_speed = int_frame.get(object_id, None)
        float_speed = float_frame.get(object_id, None)

        if int_speed != float_speed:
            current_mismatches["mismatches"].append(
                {
                    "object": object_id,
                    "int_speed": int_speed,
                    "float_speed": float_speed,
                }
            )
        difference = round(float_speed - int_speed, 2)
        report_writer.writerow(
            [
                idx + 1,
                object_id,
                int_speed,
                float_speed,
                difference if difference else "",
            ]
        )
        # table.add_row(
        #     str(idx),
        #     str(object_id),
        #     str(int_speed),
        #     str(float_speed),
        # )
    if current_mismatches["mismatches"]:
        mismatches.append(current_mismatches)

# table = rich.table.Table(title="Mismatches", show_lines=True)
# table.add_column("Frame")
# table.add_column("Object")
# table.add_column("Int Speed")
# table.add_column("Float Speed")
# table.add_column("Difference")

# for mismatch in mismatches:
#     for mismatch_data in mismatch["mismatches"]:
#         table.add_row(
#             str(mismatch["frame"]),
#             str(mismatch_data["object"]),
#             str(mismatch_data["int_speed"]),
#             str(mismatch_data["float_speed"]),
#         )

# rich.print(table)
