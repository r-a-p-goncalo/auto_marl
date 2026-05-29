from automarl.utils.visualization.mermaid.mermaid_renderer import MermaidRenderer
from automarl.utils.visualization.component_graph import Graph

class MermaidObjectDiagramRenderer(MermaidRenderer):

    PATH_TO_SAVE = "objectdiagram_mermaid.md"


    def render(self, graph):

         lines = ["classDiagram"]

         for node in graph.nodes.values():

             lines.append(self._render_node(node))

         for edge in graph.edges:

             lines.append(
                 f"{edge.source} --> {edge.target}"
             )

         return "\n".join(lines)
    

    def _render_node(self, node):
    
        lines = []
    
        lines.append(f"class {node.id} {{")
    
        lines.append(f"    {node.type_name}")
    
        for key, value in node.attributes.items():
        
            safe_value = self._escape(str(value))
    
            lines.append(
                f"    {key} = {safe_value}"
            )
    
        lines.append("}")
    
        return "\n".join(lines)