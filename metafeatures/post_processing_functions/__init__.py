import os

from metafeatures.aux.discovery import discover_components
from metafeatures.post_processing_functions.base import PostProcessing


post_processing_functions_directory = os.path.split(__file__)[0]
post_processing_functions = discover_components(
    __package__, post_processing_functions_directory, PostProcessing)