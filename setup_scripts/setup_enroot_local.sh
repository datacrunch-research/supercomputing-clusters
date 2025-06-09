#!/bin/bash

set -euo pipefail

# Function to print error messages
error_exit() {
    echo "[ERROR] $1" >&2
    exit 1
}

# Create directories and check for errors
echo "Creating directories..."
sudo mkdir -p /mnt/local_disk/enroot/data || error_exit "Failed to create /mnt/local_disk/enroot/data"
sudo mkdir -p /mnt/local_disk/enroot/cache || error_exit "Failed to create /mnt/local_disk/enroot/cache"
sudo mkdir -p /mnt/local_disk/enroot/config || error_exit "Failed to create /mnt/local_disk/enroot/config"
sudo mkdir -p /mnt/local_disk/enroot/runtime || error_exit "Failed to create /mnt/local_disk/enroot/runtime"

echo "Changing ownership..."
sudo chown -R "$USER:$USER" /mnt/local_disk/enroot || error_exit "Failed to change ownership of /mnt/local_disk/enroot"

echo "Enroot local setup complete"

# Update /etc/enroot/enroot.conf with new paths
enroot_conf="/etc/enroot/enroot.conf"
echo "Updating $enroot_conf..."

if [ ! -w "$enroot_conf" ]; then
    echo "[ERROR] $enroot_conf is not writable. Trying with sudo..."
    SUDO=sudo
else
    SUDO=""
fi

$SUDO cp "$enroot_conf" "$enroot_conf.bak" || error_exit "Failed to backup $enroot_conf"

$SUDO sed -i \
    -e 's|^#ENROOT_LIBRARY_PATH.*|ENROOT_LIBRARY_PATH        /usr/lib/enroot|' \
    -e 's|^#ENROOT_SYSCONF_PATH.*|ENROOT_SYSCONF_PATH        /etc/enroot|' \
    -e 's|^#ENROOT_RUNTIME_PATH.*|ENROOT_RUNTIME_PATH        /mnt/local_disk/enroot/runtime|' \
    -e 's|^#ENROOT_CONFIG_PATH.*|ENROOT_CONFIG_PATH         /mnt/local_disk/enroot/config|' \
    -e 's|^#ENROOT_CACHE_PATH.*|ENROOT_CACHE_PATH          /mnt/local_disk/enroot/cache|' \
    -e 's|^#ENROOT_DATA_PATH.*|ENROOT_DATA_PATH           /mnt/local_disk/enroot/data|' \
    -e 's|^#ENROOT_TEMP_PATH.*|ENROOT_TEMP_PATH           ${TMPDIR:-/tmp}|' \
    "$enroot_conf" || error_exit "Failed to update $enroot_conf"

echo "$enroot_conf updated successfully. Backup saved as $enroot_conf.bak."



