from automarl.utils.visualization.mermaid.mermaid_renderer import MermaidRenderer
from automarl.utils.visualization.component_graph import Graph, GraphEdge, GraphNode

class MermaidObjectDiagramRenderer(MermaidRenderer):

    PATH_TO_SAVE = "objectdiagram_mermaid.md"


    def _generate_base_name_to_render(self, node : GraphNode):

        if node.display_name != node.class_name:
            return f"{node.display_name}({node.class_name})"
        
        else:
            return node.display_name
        

    def _render_node_name(self, node : GraphNode):

        name_to_render = self._individual_names.get(node.id)

        if name_to_render is not None:
            return name_to_render

        name_to_render = self._generate_base_name_to_render(node)
        
        self._individual_names[node.id] = name_to_render
        return name_to_render

    def render(self, graph):
         
         self._individual_names = {}

         lines = ["classDiagram"]

         for node in graph.nodes.values():

             lines.append(self._render_node(node))

         for edge in graph.edges:

             lines.append(
                 self._render_edge(edge, graph)
             )

         return "\n".join(lines)
    

    def _render_node(self, node : GraphNode):
    
        lines = []
    
        lines.append(f"class {node.id}[\"{self._render_node_name(node)}\"] {{")
        
        for key, value in node.attributes.items():
        
            safe_value = self._escape(str(value))
    
            lines.append(
                f"    {key} = {safe_value}"
            )
    
        lines.append("}")
    
        return "\n".join(lines)
    

    def _render_edge(self, edge, graph : Graph):

        if edge.edge_type == GraphEdge.PARENT_CHILD_TYPE:

            connector = "*--"

        elif edge.edge_type == GraphEdge.REFERENCE_TYPE:

            connector = "..>"

        else:

            connector = "-->"

        label = ""

        if edge.label is not None:
            label = f" : {self._escape(edge.label)}"

        return (
            f"{edge.source} "
            f"{connector} "
            f"{edge.target}"
            f"{label}"
        )