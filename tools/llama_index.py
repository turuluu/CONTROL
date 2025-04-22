from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.tools import FunctionTool
from pathlib import Path
import random
import tempfile
import subprocess

ddg_tool_spec = DuckDuckGoSearchToolSpec()
ddg_search = FunctionTool.from_defaults(ddg_tool_spec.duckduckgo_full_search)


def weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


weather_info_tool = FunctionTool.from_defaults(weather_info)

nix_bin_path = Path.cwd() / './result/bin'


def puml_to_png(puml_declaration: str, output_path: str, plantuml_path: str = nix_bin_path / '/plantuml'):
    """
        Generates a PNG image from a UML declaration using PlantUML.

        Parameters:
        -----------
        uml_text : str
            The UML declaration in PlantUML syntax (e.g., "@startuml ... @enduml").
        output_path : str
            The desired path for the output PNG file.
        plantuml_path : str
            Path to the PlantUML executable (default is './result/bin/plantuml', e.g., from `nix build`).

        Returns:
        --------
        str
            The path to the generated PNG file.

        Raises:
        -------
        subprocess.CalledProcessError
            If the PlantUML command fails to run.
        """
    if not nix_bin_path.exists():
        print(f"Aborting puml to png conversion, binary path incorrect or missing...{nix_bin_path}")
        return ''

    # Write UML to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".puml", delete=False) as temp_file:
        temp_file.write(puml_declaration.encode("utf-8"))
        temp_file_path = temp_file.name

    # Generate PNG using PlantUML
    subprocess.run([plantuml_path, "-tpng", temp_file_path], check=True)

    # Write to final destination
    png_path = Path(temp_file_path.replace(".puml", ".png"))
    png_path.rename(output_path)
    Path(temp_file_path).unlink()

    return output_path


puml_to_png_tool = FunctionTool.from_defaults(puml_to_png)
