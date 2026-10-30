from controller import Controller, ControllerAction
from dataloader import Frame

from flask import Flask, request

from dataclasses import asdict
from pathlib import Path
import json
import sys

app = Flask(__name__)

controller = Controller(Path("./exports/finetune.pt"))

@app.post("/update/")
def update():
    result = controller.update(Frame.from_dict(request.get_json()))
    if result is None:
        return json.dumps(None)
    data = asdict(result)
    data["state"] = result.state.name
    return json.dumps(data)

@app.post("/reset/")
def reset():
    x = float(request.args["x"])
    y = float(request.args["y"])
    controller.reset(x, y)

    return json.dumps(None), 204

if __name__ == "__main__":
    port = int(sys.argv[1])
    app.run(port=port, threaded=False, debug=False)
