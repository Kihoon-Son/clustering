import json
import networkx as nx
from numpy import genfromtxt


def read_json(all_fp_dir):
    # Use a breakpoint in the code line below to debug your script.
    with open(all_fp_dir) as json_file:
        json_data = json.load(json_file)
    print(len(json_data))
    return json_data


def find(fp_name, all_fp_json):
    for json_data in all_fp_json:
        if fp_name == json_data["name"]:
            return json_data


def read_csv(dir):
    return genfromtxt(dir, delimiter=',')


#Calculate similarities
def number_similarity(user_number_data, reference_number_data):
  count = 0
  for i in range(5):
    if user_number_data['number'][i] == reference_number_data['number'][i]:
      count += 1
  return count/5


def overallshape_similarity(user_overallshape_data, reference_overallshape_data):
  count = 0
  for i in range(64):
    if user_overallshape_data['grids'][i] == reference_overallshape_data['grids'][i]:
        count += 1
  # considering same aspect ratio
  # if((user_overallshape_data['aspect'] - reference_overallshape_data['aspect']) == 0):
  return ((1-count/64) + (abs(user_overallshape_data['aspect'] - reference_overallshape_data['aspect'])))/2


# def location_similarity(user_location_data, reference_location_data):
#   user_location_graph = nx.readwrite.node_link_graph(user_location_data)
#   referen_location_graph = nx.readwrite.node_link_graph(reference_location_data)
#   ged = gm.GraphEditDistance(1,1,1,1) # all edit costs are equal to 1
#   locationResult = ged.compare([user_location_graph,referen_location_graph],None)
#   return locationResult
#
# def connectivity_similarity(user_connectivity_data, reference_connectivity_data):
#   user_connectivity_graph = nx.readwrite.node_link_graph(user_connectivity_data)
#   referen_connectivity_graph = nx.readwrite.node_link_graph(reference_connectivity_data)
#   ged=gm.GraphEditDistance(1,1,1,1) # all edit costs are equal to 1
#   connectivityResult = ged.compare([user_connectivity_graph,referen_connectivity_graph],None)
#   return connectivityResult