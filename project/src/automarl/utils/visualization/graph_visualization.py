
from automarl.component import Component
from automarl.utils.visualization.component_graph_extractor import ComponentGraphExtractor
from automarl.utils.visualization.component_graph import Graph
from automarl.utils.visualization.component_graph_renderer import GraphRenderer
import os

def extract_graph(component : Component, extractor : ComponentGraphExtractor):
    
    graph = extractor.extract([component])
    return graph


def render_component_graph(
    component_graph: Graph,
    renderer : GraphRenderer,
    path
    ):


    return renderer.render_and_save_to_file(component_graph, path)

def render_component_system(component : Component,
                            extractor : ComponentGraphExtractor,
                            renderer : GraphRenderer,
                            output_path: str = None):
    
    if isinstance(extractor, type):
        extractor : ComponentGraphExtractor = extractor()
    
    if isinstance(renderer, type):
        renderer : GraphRenderer = renderer()
    
    graph = extract_graph(component, extractor)

    if output_path is None:
            component : ArtifactComponent = component
            output_path = os.path.join(f"{component.get_artifact_directory()}")

    rendered_graph = render_component_graph(graph, renderer, component.get_artifact_directory())
    
    return rendered_graph

