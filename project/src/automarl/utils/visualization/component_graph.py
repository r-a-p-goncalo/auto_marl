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


    

