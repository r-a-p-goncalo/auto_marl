from dataclasses import dataclass, field
from typing import Any

from automarl.component import Component



@dataclass
class GraphNode:
    
    id: str

    display_name: str

    class_name: str
    
    attributes: dict[str, Any] = field(default_factory=dict)
    
    data: Any = None


@dataclass
class GraphEdge:

    PARENT_CHILD_TYPE = "parent_child"
    REFERENCE_TYPE = "reference"

    source: str
    target: str
    label: str | None = None
    edge_type: str = REFERENCE_TYPE

    def change_label(self, new_label):
        self.label = new_label


@dataclass
class Graph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode):
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge):
        self.edges.append(edge)

    def get_node(self, id):
        return self.nodes.get(id)
    
    def get_edges(self, source_node_id : str, target_node_id : str, check_both_ways = False):

        to_return = []

        for edge in self.edges:

            if edge.source == source_node_id and edge.target == target_node_id:
                to_return.append(edge)
            
            elif check_both_ways and edge.source == target_node_id and edge.target == source_node_id:
                to_return.append(edge)

        return to_return
    
    def remove_edge(self, edge_to_remove : GraphEdge):
        self.edges.remove(edge_to_remove)

    

