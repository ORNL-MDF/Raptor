import argparse
import json
import logging
import os
import sys
from pathlib import Path

from intersect_sdk import (
    IntersectService,
    IntersectServiceConfig,
    default_intersect_lifecycle_loop,
)

logger = logging.getLogger(__name__)

"""
Launch the service separately due to its module and import structure.
"""

if __name__ == "__main__":
    # boilerplate config file setup
    parser = argparse.ArgumentParser(description="Automated client")
    parser.add_argument(
        "--config",
        type=Path,
        default=os.environ.get(
            "DIAL_CONFIG_FILE", Path(__file__).parents[1] / "local-conf.json"
        ),
    )
    args = parser.parse_args()
    try:
        with Path(args.config).open("rb") as f:
            from_config_file = json.load(f)
    except (json.decoder.JSONDecodeError, OSError) as e:
        logger.critical("unable to load config file: %s", str(e))
        sys.exit(1)

    config = IntersectServiceConfig(
        hierarchy=from_config_file["intersect-hierarchy"],
        **from_config_file["intersect"],
    )

    logging.basicConfig(level=logging.INFO)

    # IMPORTANT: import this after logging configuration
    from dial_service import DialCapabilityImplementation

    capability = DialCapabilityImplementation(from_config_file["dial"]["mongo"])

    """
    Step three: create the service from its configuration and capability.
    """
    service = IntersectService([capability], config)

    """
    Step four: start the lifecycle loop with the service.
    Some applications, such as REST APIs, should integrate the service into
    their existing lifecycle. In that case, call service.startup() and
    service.shutdown() at the appropriate stages.
    """
    default_intersect_lifecycle_loop(
        service,
    )

    """
    The service runs until the application is explicitly stopped (e.g. Ctrl+C).
    """
