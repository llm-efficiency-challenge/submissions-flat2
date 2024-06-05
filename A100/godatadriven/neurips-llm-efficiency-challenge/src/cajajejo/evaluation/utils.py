import logging
import typing

import sh

logger = logging.getLogger("cajajejo.evaluation.utils")


def _parse_helm_header(record):
    """Parse HELM result headers"""
    return [v["value"] for v in record]


def _parse_helm_values(record):
    """Parse HELM result values"""
    return [v.get("value") for v in record]


def run_sh_with_exception_handling(cmd: sh.Command, args: typing.List[str]):
    """Run a shell command with exception handling"""
    try:
        print(cmd(args))
    except sh.ErrorReturnCode as e:
        print(e.stderr.decode("utf-8"))
        logger.debug(e.stderr.decode("utf-8"))
        logger.debug(f"Command failed with exit code={e.exit_code}")
        raise e
