# importing the required packages and modules
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from neo4j import GraphDatabase
import dgl
import h5py
import pickle

# creating the flask app
app = Flask(__name__)

with open('graph_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

nodes = loaded_data['nodes']
edges = loaded_data['edges']
features_dict = loaded_data['features_dict']
author_id_dict = loaded_data['author_id_dict']

src, dst = zip(*edges)
src = np.array(src)
dst = np.array(dst)
node_id_to_dgl_id = {node_id: dgl_id for dgl_id, node_id in enumerate(nodes)}
all_nodes = list(node_id_to_dgl_id.values())
g = dgl.graph((src, dst), num_nodes=len(all_nodes))

# Loading the saved data from the model.h5 file
model_path = 'model.h5'
recommended_nodes_dict = {}
likelihood_scores_dict = {}

# Reading data from the file and storing it as dictionaries
with h5py.File(model_path, 'r') as f:
    recommended_nodes_data = f['recommended_nodes'][:]
    likelihood_scores_data = f['likelihood_scores'][:]

    for idx, node_id in enumerate(all_nodes):
        recommended_nodes_dict[node_id] = recommended_nodes_data[idx]
        likelihood_scores_dict[node_id] = likelihood_scores_data[idx]


@app.route('/')
def index():
    return render_template('index.html')

# Adding a route to serve static files in our case style.css
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Function to get the recommendations
@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    try:
        # Parse the author ID from the query parameter 'id' as mentioned in the GET request format
        author_id = request.args.get('id')
        print(author_id)
        # Finding the corresponding node ID for the given author ID
        node_id = None
        for id, a_id in author_id_dict.items():
            if a_id == author_id:
                node_id = id
                break

        if node_id is None:
            return jsonify({"error": "Author ID not found."})

        print(node_id)
        # Finding the recommended nodes and likelihood scores from the saved data
        recommended_nodes = recommended_nodes_dict.get(node_id, [])
        print(recommended_nodes)
        likelihood_scores = likelihood_scores_dict.get(node_id, [])
        print(likelihood_scores)

        # Converting recommended_nodes to DGL IDs before using them as keys
        recommended_dgl_ids = [node_id_to_dgl_id[recommended_node] for recommended_node in recommended_nodes]

        # Converting likelihood_scores to a serializable format (Python list)
        likelihood_scores_list = likelihood_scores.tolist()

        # Preparing the response in the required format
        recommendations = []
        for rank, (recommended_node, likelihood) in enumerate(zip(recommended_dgl_ids, likelihood_scores_list), start=1):
            recommended_author_id = author_id_dict[recommended_node]
            recommendation = {
                "authorID": recommended_author_id,
                "likeliness": likelihood,
                "rank": rank
            }
            recommendations.append(recommendation)

        return jsonify(recommendations)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
