# server.py
from flask import Flask, request, render_template, send_from_directory
from nodulereconstruction_v0_4 import Nodule_Reconstruction
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('Demonstrator.html')

@app.route('/run', methods=['POST'])
def run_reconstruction():
    dcm_path = request.form['dcm_path']
    json_path = request.form['json_path']

    try:
        reconstructor = Nodule_Reconstruction(dcm_path, json_path)
        volume = reconstructor.build_3d_Mask()
        reconstructor.visualize_3d_Mask(volume)
        reconstructor.compute_max_nodule_extent(volume)
        return render_template('Demonstrator.html')
    except Exception as e:
        return f"<h2>Error occurred:</h2><pre>{str(e)}</pre>"

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

import webbrowser
import threading
import os

if __name__ == '__main__':
    port = 5000
    url = f'http://localhost:{port}'

    # Only open browser if not inside Flask reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(debug=True, port=port)
