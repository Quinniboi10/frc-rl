from controller import Controller
from dataloader import Frame

from flask import Flask, request

from dataclasses import asdict
from pathlib import Path
import logging
import json
import sys

app = Flask(__name__)

controllers: dict[int, Controller] = {}

@app.post("/update/<int:request_id>/")
def update(request_id: int):
    result = controllers[request_id].update(Frame.from_dict(request.get_json()))
    if result is None:
        return json.dumps(None)
    data = asdict(result)
    data["state"] = result.state.name
    return json.dumps(data)

@app.post("/reset/<int:request_id>/")
def reset(request_id: int):
    if not request_id in controllers:
        controllers[request_id] = Controller(Path("./exports/finetune.pt"))
        logging.info(f"Registered new controller with ID {request_id}")

    args = request.get_json()
    x = float(args["x"])
    y = float(args["y"])
    controllers[request_id].reset(x, y)

    return "", 204

if __name__ == "__main__":
    port = int(sys.argv[1])
    app.run(port=port, threaded=True, debug=False)
