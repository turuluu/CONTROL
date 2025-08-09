from typing import Optional
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.tools import FunctionTool
import llama_index.core
from pathlib import Path

from persistent_cache.decorators import persistent_cache
import mediawiki

import random
import tempfile
import subprocess
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

wiki = mediawiki.MediaWiki()

llama_index.core.set_global_handler("simple")

@persistent_cache(hours=1)
def wikipedia_search(search_terms: str) -> str:
    """Searches wikipedia for encyclopedic information about topics"""
    print(f'[tool] Searching wikipedia {search_terms=}')
    response = ''
    for wiki_resp in wiki.search(search_terms)[:1]:
        try:
            response += wiki.page(wiki_resp).wikitext
        except mediawiki.exceptions.DisambiguationError:
            pass
    return response


wikipedia_tool = FunctionTool.from_defaults(wikipedia_search)


@persistent_cache(hours=1)
def ddg_search(query: str) -> str:
    """
    Make a query to DuckDuckGo search to receive a full search results.

    Args:
        query (str): The query to be passed to DuckDuckGo.
    """
    from duckduckgo_search import DDGS

    params = {
        "keywords": query,
        "region": 'wt-wt',
        "max_results": 5,
    }

    with DDGS() as ddg:
        return list(ddg.text(**params))


# web search
ddg_tool_spec = DuckDuckGoSearchToolSpec()
web_search_tool = FunctionTool.from_defaults(ddg_search)
ddg_search = web_search_tool


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

nix_bin_path = Path.cwd() / 'result/bin'


def puml_to_png(puml_declaration: str, output_path: str, plantuml_path: str = nix_bin_path / 'plantuml') -> Optional[
    str]:
    """
        Generates a PNG image from a UML declaration using PlantUML.

        Args:
            uml_text (str): The UML declaration in PlantUML syntax (e.g., "@startuml ... @enduml").
            output_path (str): The desired path for the output PNG file.
            plantuml_path (str): Path to the PlantUML executable (default is './result/bin/plantuml', e.g., from `nix build`).

        Returns:
            Optional[str]: The path to the generated PNG file.

        Raises:
            subprocess.CalledProcessError: If the PlantUML command fails to run.
        """
    if not nix_bin_path.exists():
        print(f"Aborting puml to png conversion, binary path incorrect or missing: {nix_bin_path}")
        print('Look for build commands for puml demo in the readme')
        return

    if not puml_declaration:
        print('puml empty')
        return

    # Write UML
    with tempfile.NamedTemporaryFile(suffix=".puml", delete=False) as temp_file:
        temp_file.write(puml_declaration.encode("utf-8"))
        temp_file_path = temp_file.name

    # Generate PNG
    subprocess.run([plantuml_path, "-tpng", temp_file_path], check=True)

    # Write to final destination
    png_path = Path(temp_file_path.replace(".puml", ".png"))
    png_path.rename(output_path)
    Path(temp_file_path).unlink()

    return output_path


puml_to_png_tool = FunctionTool.from_defaults(puml_to_png)
