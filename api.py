import os
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify

# Set environment variables
MODEL_DIR = "../aicoe-osc-demo/models/distilbert_mnli_pruned80/"
MODEL_FILE = "model.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Loading model
print("Loading model from: {}".format(MODEL_PATH))
inference = ort.InferenceSession(MODEL_PATH)

# Creation of the Flask app
app = Flask(__name__)

# API 
# Flask route so that we can serve HTTP traffic on that route
@app.route('/', methods=['POST'])
def prediction():
    # Get input data from request
    input_data = request.get_json()

    # Validate input data
    if input_data is None or "data" not in input_data:
        return jsonify({"error": "Invalid input data"})

    # Run inference on input data
    input_array = np.array(input_data["data"], dtype=np.float32)
    output_array = inference.run(None, {"input": input_array})[0]

    # Return prediction as response
    return jsonify({"prediction": output_array.tolist()})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000) # Launch built-in we server and run this Flask webapp


