import json


PREDS_FILE = "test_preds.json"
ORDER = "filenames_order.json"
with open(PREDS_FILE, "r") as preds_file:
    with open(ORDER, "r") as order_file:
        preds = json.load(preds_file)
        classes = preds["pred_class"]

        names = json.load(order_file)


print(names[:5])
print(classes[:5])

DST = "pred.txt"
with open(DST, "wt") as dst_file:
    assert len(names) == len(classes)
    for name, cls in zip(names, classes):
        dst_file.write(f"{name} {cls}\n")
