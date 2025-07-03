#!/bin/bash
set -e  # Exit on error
#set -x  # Uncomment to print each command as it runs

echo "=== Enroot Health Check ==="

# Check if enroot is installed
echo "Checking if enroot is installed..."
if ! command -v enroot; then
    echo "ERROR: Enroot is not installed or not in PATH."
    exit 1
fi

# Check enroot version
echo "Checking enroot version..."
if ! enroot version; then
    echo "ERROR: Enroot is installed but not functioning correctly."
    exit 1
fi

echo "Enroot version: $(enroot version)"

# Try to import and start a minimal container
TEST_IMG="ubuntu"
TEST_SQSH="ubuntu.sqsh"
TEST_CONT="enroot_healthcheck"

# Clean up from previous runs
echo "Cleaning up previous test artifacts..."

if enroot list | grep -q $TEST_CONT; then
    enroot remove -f $TEST_CONT
    echo "Removed existing container: $TEST_CONT"
else
    echo "No existing container ($TEST_CONT) to remove."
fi

if [ -f $TEST_SQSH ]; then
    rm -f $TEST_SQSH
    echo "Removed existing squashfs file: $TEST_SQSH"
else
    echo "No squashfs file ($TEST_SQSH) to remove."
fi

# Import the test image
echo "Importing test image..."
if ! enroot import docker://$TEST_IMG; then
    echo "ERROR: Failed to import docker://$TEST_IMG"
    exit 1
fi

# Create a test container from the imported image
echo "Creating test container..."
if ! enroot create -n $TEST_CONT $TEST_SQSH; then
    echo "ERROR: Failed to create container from $TEST_SQSH"
    rm -f $TEST_SQSH
    exit 1
fi

# Start the test container
echo "Checking PID of the container..."
PID_OUTPUT=$(enroot start $TEST_CONT sh -c 'echo $$')
OS_OUTPUT=$(enroot start $TEST_CONT sh -c 'grep PRETTY /etc/os-release')

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start test container"
    enroot remove -f $TEST_CONT || true
    rm -f $TEST_SQSH
    exit 1
fi
echo "PID inside the container: $PID_OUTPUT"
echo "OS information: $OS_OUTPUT"

echo "Enroot health check PASSED."

# Clean up
echo "Cleaning up test artifacts..."
enroot remove -f $TEST_CONT || true
rm -f $TEST_SQSH