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
This launches the service.  Separate file due to module/import structure, plus we possibly want the capability to be a separate unit
"""

if __name__ == '__main__':
    # boilerplate config file setup
    parser = argparse.ArgumentParser(description='Automated client')
    parser.add_argument(
        '--config',
        type=Path,
        default=os.environ.get('DIAL_CONFIG_FILE', Path(__file__).parents[1] / 'local-conf.json'),
    )
    args = parser.parse_args()
    try:
        with Path(args.config).open('rb') as f:
            from_config_file = json.load(f)
    except (json.decoder.JSONDecodeError, OSError) as e:
        logger.critical('unable to load config file: %s', str(e))
        sys.exit(1)

    config = IntersectServiceConfig(
        hierarchy=from_config_file['intersect-hierarchy'],
        **from_config_file['intersect'],
    )

    logging.basicConfig(level=logging.INFO)

    # IMPORTANT: import this after logging configuration
    from dial_service import DialCapabilityImplementation

    capability = DialCapabilityImplementation(from_config_file['dial']['mongo'])

    """
    step three - create service from both the configuration and your own capability
    """
    service = IntersectService([capability], config)

    """
    step four - start lifecycle loop. The only necessary parameter is your service.
    with certain applications (i.e. REST APIs) you'll want to integrate the service in the existing lifecycle,
    instead of using this one.
    In that case, just be sure to call service.startup() and service.shutdown() at appropriate stages.
    """
    default_intersect_lifecycle_loop(
        service,
    )

    """
    Note that the service will run forever until you explicitly kill the application (i.e. Ctrl+C)
    """
