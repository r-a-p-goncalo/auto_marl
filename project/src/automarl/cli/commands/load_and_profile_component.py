import cProfile
import pstats
import argparse
import os

from automarl.cli.load_run_component import generate_path, load_component, run_component  

def profiled_main(component_path, target_dir=None, target_dir_name=None, output_file="profile.out"):

    if target_dir is not None or target_dir_name is not None:
        component_path = generate_path(component_path, target_dir, target_dir_name)
    
    output_file = os.path.join(component_path, output_file)

    # Run with profiling
    profiler = cProfile.Profile()
    profiler.enable()

    component = load_component(component_path)
    run_component(component)

    profiler.disable()

    # Dump stats
    profiler.dump_stats(output_file)

    print(f"Profile saved to: {os.path.abspath(output_file)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run component with cProfile.")
    
    parser.add_argument("--component_path", type=str, default='.', help="Path of experiment")
    parser.add_argument("--target_dir", type=str, default=None, help="Directory where to put the component run")
    parser.add_argument("--target_dir_name", type=str, default=None, help="Last folder where to put the run")
    parser.add_argument("--output", type=str, default="profile.out", help="Output profile file")

    args = parser.parse_args()

    profiled_main(
        component_path=args.component_path,
        target_dir=args.target_dir,
        target_dir_name=args.target_dir_name,
        output_file=args.output
    )
