#!/usr/bin/env python
"""
Test Runner Script for NextRec

This script provides a convenient way to run tests with various options.
"""
import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_tests(test_type="all", verbose=True, coverage=False, markers=None):
    """
    Run tests with specified options

    Args:
        test_type: Type of tests to run ('all', 'match', 'ranking', 'multitask')
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        markers: Pytest markers to filter tests

    Returns:
        int: Exit code from pytest
    """
    cmd = [sys.executable, "-m", "pytest"]

    # Determine which tests to run
    if test_type == "all":
        cmd.append("test/")
    elif test_type == "match":
        cmd.append("test/test_match_models.py")
    elif test_type == "ranking":
        cmd.append("test/test_ranking_models.py")
    elif test_type == "multitask":
        cmd.append("test/test_multitask_models.py")
    else:
        logger.error(f"Unknown test type: {test_type}")
        return 1

    # Add verbose flag
    if verbose:
        cmd.extend(["-v", "-s"])

    # Add coverage
    if coverage:
        cmd.extend(["--cov=nextrec", "--cov-report=html", "--cov-report=term-missing"])

    # Add markers
    if markers:
        cmd.extend(["-m", markers])

    # Log command
    logger.info(f"Running command: {' '.join(cmd)}")

    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        logger.warning("Tests interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run NextRec unit tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py
  
  # Run match model tests only
  python run_tests.py --type match
  
  # Run with coverage
  python run_tests.py --coverage
  
  # Run specific test pattern
  python run_tests.py --type ranking -k "deepfm"
  
  # Run without verbose output
  python run_tests.py --quiet
        """,
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["all", "match", "ranking", "multitask"],
        default="all",
        help="Type of tests to run (default: all)",
    )

    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Generate coverage report"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output verbosity"
    )

    parser.add_argument(
        "--markers", "-m", type=str, help="Run tests matching given mark expression"
    )

    args = parser.parse_args()

    # Display banner
    logger.info("=" * 80)
    logger.info("NextRec Unit Test Runner")
    logger.info("=" * 80)
    logger.info(f"Test type: {args.type}")
    logger.info(f"Coverage: {args.coverage}")
    logger.info(f"Verbose: {not args.quiet}")
    if args.markers:
        logger.info(f"Markers: {args.markers}")
    logger.info("=" * 80)

    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        verbose=not args.quiet,
        coverage=args.coverage,
        markers=args.markers,
    )

    # Summary
    logger.info("=" * 80)
    if exit_code == 0:
        logger.info("✓ All tests passed!")
    else:
        logger.error(f"✗ Tests failed with exit code {exit_code}")
    logger.info("=" * 80)

    if args.coverage and exit_code == 0:
        logger.info("Coverage report generated in htmlcov/index.html")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
