from automarl.utils.visualization.component_graph import Graph
import os

class GraphRenderer:

    PATH_TO_SAVE = "graph.md"


    def render(self, graph: Graph) -> str:
        raise NotImplementedError
    
    def _save_render_to_file(self, rendered_graph, path_to_save):
        pass

    def save_render_to_file(self, rendered_graph, path_to_save, graph_file_name=None):

        if graph_file_name is None:
            graph_file_name = self.PATH_TO_SAVE

        self._save_render_to_file(rendered_graph, path_to_save=os.path.join(path_to_save, graph_file_name))

    def render_and_save_to_file(self, graph, path_to_save, graph_file_name=None):

        rendered_graph = self.render(graph)

        self.save_render_to_file(rendered_graph, path_to_save, graph_file_name)

        return rendered_graph

